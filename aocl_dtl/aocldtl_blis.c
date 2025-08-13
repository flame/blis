/*===================================================================
 * File Name :  aocldtl_blis.c
 *
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/

#include "blis.h"

dim_t AOCL_get_requested_threads_count(void)
{
    dim_t nthreads = bli_thread_get_num_threads();
    // If BLIS ways parallelism has been set, or if the OpenMP level
    // is not active, then stored nt is currently -1. Change value to
    // 1 for printing in logs
    if ( nthreads < 0 ) nthreads = 1;
    return nthreads;
}

#if AOCL_DTL_LOG_ENABLE

// Helper functions

void DTL_get_complex_parts(char dt_type,
                           const void *complex_input,
                           double *real,
                           double *imag)
{
    if (dt_type == 'S' || dt_type == 's')
    {
        *real = *((float *)complex_input);
        *imag = 0.0;
    }
    else if (dt_type == 'D' || dt_type == 'd')
    {
        *real = *((double *)complex_input);
        *imag = 0.0;
    }
    else if (dt_type == 'c' || dt_type == 'C')
    {
        *real = (float)(((scomplex *)complex_input)->real);
        *imag = (float)(((scomplex *)complex_input)->imag);
    }
    else if (dt_type == 'z' || dt_type == 'Z')
    {
        *real = ((dcomplex *)complex_input)->real;
        *imag = ((dcomplex *)complex_input)->imag;
    }
}

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
                             int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // Ordering as per cblas/blas interfaces
    // {S, D, C, Z} transa, transb, m, n, k, alpha_real, alpha_imag,
    //              lda, ldb, beta_real, beta_imag, ldc
    sprintf(buffer, "%c %c %c %ld %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            tolower(dt_type),
            transa, transb,
            (dim_t)m, (dim_t)n, (dim_t)k,
            alpha_real, alpha_imag,
            (inc_t)lda, (inc_t)ldb,
            beta_real, beta_imag,
            (inc_t)ldc);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_gemm_stats(int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k)
{
    char buffer[256];

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

    double flops = 2.0 * m * n * k;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 4.0 * flops;
    }

    if (execution_time != 0.0)
        sprintf(buffer, " nt=%ld %.3f ms %0.3f GFLOPS",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0,
                flops/(execution_time * 1e3));
    else
        sprintf(buffer, " nt=%ld %.3f ms",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0);

    DTL_Trace(loglevel, TRACE_TYPE_RAW, NULL, NULL, 0, buffer);
}

void AOCL_DTL_log_gemmt_sizes(int8 loglevel,
                              char dt_type,
                              char uplo,
                              char transa,
                              char transb,
                              const f77_int n,
                              const f77_int k,
                              const void *alpha,
                              const f77_int lda,
                              const f77_int ldb,
                              const void *beta,
                              const f77_int ldc,
                              const char *filename,
                              const char *function_name,
                              int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;


    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S,D,C,Z} {triangC : l or u} {n k lda ldb ldc transa transb alpha_real alpha_imaginary
    // beta_real, beta_imaginary}
    sprintf(buffer, "%c %c %ld %ld %lu %lu %lu %c %c %lf %lf %lf %lf",
            tolower(dt_type), uplo, (dim_t)n, (dim_t)k,
            (dim_t)lda, (dim_t)ldb, (dim_t)ldc,
            transa, transb,
            alpha_real, alpha_imag,
            beta_real, beta_imag);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_gemmt_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int k)
{
    char buffer[256];

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

    double flops = n * n * k;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 4.0 * flops;
    }

    if (execution_time != 0.0)
        sprintf(buffer, " nt=%ld %.3f ms %0.3f GFLOPS",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0,
                flops/(execution_time * 1e3));
    else
        sprintf(buffer, " nt=%ld %.3f ms",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0);

    DTL_Trace(loglevel, TRACE_TYPE_RAW, NULL, NULL, 0, buffer);
}

void AOCL_DTL_log_hemm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_int m,
                             const f77_int n,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {C, Z} { side, uploa, m, n, alpha_real, alpha_imag, lda, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
            tolower(dt_type), side, uploa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag,
            (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_her2k_sizes(int8 loglevel,
                              char dt_type,
                              const f77_char uploc,
                              const f77_char transa,
                              const f77_int m,
                              const f77_int k,
                              const void *alpha,
                              const f77_int lda,
                              const f77_int ldb,
                              const void *beta,
                              const f77_int ldc,
                              const char *filename,
                              const char *function_name,
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {C, Z} { uploc, transa, m, k, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_herk_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploc,
                             const f77_char transa,
                             const f77_int m,
                             const f77_int k,
                             const void *alpha,
                             const f77_int lda,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {C, Z} {uploc, transa, m, k, alpha_real, alpha_imag, lda, beta_real, beta_imag, ldc}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %lf %lf %ld\n", tolower(dt_type),
            uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_symm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_int m,
                             const f77_int n,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D, C, Z} { side, uploa, m, n, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            side, uploa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_syr2k_sizes(int8 loglevel,
                              char dt_type,
                              const f77_char uploc,
                              const f77_char transa,
                              const f77_int m,
                              const f77_int k,
                              const void *alpha,
                              const f77_int lda,
                              const f77_int ldb,
                              const void *beta,
                              const f77_int ldc,
                              const char *filename,
                              const char *function_name,
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // { uploc, transa, m, k, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_syrk_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploc,
                             const f77_char transa,
                             const f77_int m,
                             const f77_int k,
                             const void *alpha,
                             const f77_int lda,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { uploc, transa, m, k, alpha_real, alpha_imag, lda, beta_real, beta_imag, ldc}
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %lf %lf %ld\n", tolower(dt_type),
            uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trmm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int n,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D,C, Z} { side, uploa, transa, diaga, m, n, alpha_real, alpha_imag, lda, ldb}
    sprintf(buffer, "%c %c %c %c %c %ld %ld %lf %lf %ld %ld\n", tolower(dt_type),
            side, uploa, transa, diaga, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trsm_sizes(int8 loglevel,
                             char dt_type,
                             f77_char side,
                             f77_char uploa,
                             f77_char transa,
                             f77_char diaga,
                             const f77_int m,
                             const f77_int n,
                             const void *alpha,
                             f77_int lda,
                             f77_int ldb,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    //{S, D, C, Z} side, uplo, transa, diaga, m, n, lda, ldb, alpha_real, alpha_imag
    sprintf(buffer, "%c %c %c %c %c %ld %ld %ld %ld %lf %lf", tolower(dt_type),
            side, uploa, transa, diaga,
            (dim_t)m, (dim_t)n, (dim_t)lda, (dim_t)ldb,
            alpha_real, alpha_imag);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trsm_stats(int8 loglevel,
                             char dt_type,
                             f77_char side,
                             const f77_int m,
                             const f77_int n)
{
    char buffer[256];

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

    double flops = 0.0;
    if (side == 'L' || side =='l')
    {
        flops = 1.0 * m * n * m;
    }
    else
    {
        flops = 1.0 * m * n * n;
    }
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 4.0 * flops;
    }

    if (execution_time != 0.0)
        sprintf(buffer, " nt=%ld %.3f ms %0.3f GFLOPS",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0,
                flops/(execution_time * 1e3));
    else
        sprintf(buffer, " nt=%ld %.3f ms",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0);

    DTL_Trace(loglevel, TRACE_TYPE_RAW, NULL, NULL, 0, buffer);
}

// Level-3 Extension Logging

void AOCL_DTL_log_gemm_get_size_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char identifier,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];

    // Ordering as per cblas/blas interfaces
    // {S, D, C, Z} identifier, m, n, k
    sprintf(buffer, "%c %c %ld %ld %ld\n", tolower(dt_type),
            identifier, (dim_t)m, (dim_t)n, (dim_t)k);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}


void AOCL_DTL_log_gemm_pack_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char identifier,
                             const f77_char trans,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const void *alpha,
                             const f77_int pld,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];

    // Ordering as per cblas/blas interfaces
    // {S, D, C, Z} identifier, trans, m, n, k, pld
    sprintf(buffer, "%c %c %ld %ld %ld %ld\n", tolower(dt_type),
            identifier, (dim_t)m, (dim_t)n, (dim_t)k, (dim_t)pld);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
                             int line)
{
    char buffer[256];

    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // Ordering as per cblas/blas interfaces
    // {S, D, C, Z} transa, transb, m, n, k,
    //              lda, ldb, beta_real, beta_imag, ldc
    sprintf(buffer, "%c %c %c %ld %ld %ld %ld %ld %lf %lf %ld",
            tolower(dt_type),
            transa, transb,
            (dim_t)m, (dim_t)n, (dim_t)k,
            (inc_t)lda, (inc_t)ldb,
            beta_real, beta_imag,
            (inc_t)ldc);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-2 Logging

void AOCL_DTL_log_gemv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_int m,
                             const f77_int n,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { transa, m, n, alpha, lda, incx, beta, incy}
    sprintf(buffer, "%c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            transa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag,
            (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_ger_sizes(int8 loglevel,
                            char dt_type,
                            const f77_int m,
                            const f77_int n,
                            const void *alpha,
                            const f77_int incx,
                            const f77_int incy,
                            const f77_int lda,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    sprintf(buffer, "%c %ld %ld %lf %lf %ld %ld %ld\n", tolower(dt_type),
            (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_hemv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, lda, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_her2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const f77_int lda,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D, C, Z} {uploa, m, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_her_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int m,
                            const void *alpha,
                            const f77_int incx,
                            const f77_int lda,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {C, Z} {uploa, m alpha_real, alpha_imag incx lda}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_symv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_d = 0.0;
    double beta_d = 0.0;

    if (dt_type == 's' || dt_type == 'S')
    {
        alpha_d = *((float *)alpha);
        beta_d = *((float *)beta);
    }
    else if (dt_type == 'd' || dt_type == 'D')
    {
        alpha_d = *((double *)alpha);
        beta_d = *((double *)beta);
    }

    // {S, D} { uploa, m, alpha_d, lda, incx, beta_d, incy}
    sprintf(buffer, "%c %c %ld %lf %ld %ld %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_d, (dim_t)lda, (dim_t)incx, beta_d, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_syr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const f77_int lda,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // { uploa, m, alpha_real, alpha_imag, incx, incy, lda}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}
void AOCL_DTL_log_syr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int m,
                            const void *alpha,
                            const f77_int incx,
                            const f77_int lda,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, incx, lda}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int lda,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, lda, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int lda,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, lda, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-2 Banded Logging

void AOCL_DTL_log_gbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_int m,
                             const f77_int n,
                             const f77_int kl,
                             const f77_int ku,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { transa, m, n, kl, ku, alpha, lda, incx, beta, incy}
    sprintf(buffer, "%c %c %ld %ld %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            transa, (dim_t)m, (dim_t)n, (dim_t)kl, (dim_t)ku, alpha_real, alpha_imag,
            (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_hbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const f77_int k,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { uploa, m, k, alpha_real, alpha_imag, lda, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_sbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const f77_int k,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_d = 0.0;
    double beta_d = 0.0;

    if (dt_type == 's' || dt_type == 'S')
    {
        alpha_d = *((float *)alpha);
        beta_d = *((float *)beta);
    }
    else if (dt_type == 'd' || dt_type == 'D')
    {
        alpha_d = *((double *)alpha);
        beta_d = *((double *)beta);
    }

    // {S, D} { uploa, m, k, alpha_d, lda, incx, beta_d, incy}
    sprintf(buffer, "%c %c %ld %ld %lf %ld %ld %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, (dim_t)k, alpha_d, (dim_t)lda, (dim_t)incx, beta_d, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_tbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int n,
                             const f77_int k,
                             const f77_int lda,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, n, k, lda, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)n, (dim_t)k, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_tbsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int n,
                             const f77_int k,
                             const f77_int lda,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, n, k, lda, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)n, (dim_t)k, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-2 Packed Logging

void AOCL_DTL_log_hpmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %lf %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_hpr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D, C, Z} {uploa, m, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_hpr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int m,
                            const void *alpha,
                            const f77_int incx,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {C, Z} {uploa, m alpha_real, alpha_imag incx}
    sprintf(buffer, "%c %c %ld %lf %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_spmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const void *beta,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_d = 0.0;
    double beta_d = 0.0;

    if (dt_type == 's' || dt_type == 'S')
    {
        alpha_d = *((float *)alpha);
        beta_d = *((float *)beta);
    }
    else if (dt_type == 'd' || dt_type == 'D')
    {
        alpha_d = *((double *)alpha);
        beta_d = *((double *)beta);
    }

    // {S, D} { uploa, m, alpha_d, incx, beta_d, incy}
    sprintf(buffer, "%c %c %ld %lf %ld %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_d, (dim_t)incx, beta_d, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_spr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int m,
                             const void *alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // { uploa, m, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}
void AOCL_DTL_log_spr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int m,
                            const void *alpha,
                            const f77_int incx,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, incx}
    sprintf(buffer, "%c %c %ld %lf %lf %ld\n", tolower(dt_type),
            uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_tpmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)m, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_tpsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, incx}
    sprintf(buffer, "%c %c %c %c %ld %ld\n", tolower(dt_type),
            uploa, transa, diaga, (dim_t)m, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-2 plane rotations and modified Givens transformation Logging

void AOCL_DTL_log_rot_sizes(int8 loglevel,
                            char dt_type,
                            const f77_int m,
                            const f77_int incx,
                            const f77_int incy,
                            const void *c,
                            const void *s,
                            const char *filename,
                            const char *function_name,
                            int line)
{
    char buffer[256];
    // {S, D,C, Z} {m, incx, incy, c, s}

    double c_real = 0.0;
    double c_imag = 0.0;
    double s_real = 0.0;
    double s_imag = 0.0;

    DTL_get_complex_parts(dt_type, c, &c_real, &c_imag);
    DTL_get_complex_parts(dt_type, s, &s_real, &s_imag);

    sprintf(buffer, "%c %ld %ld %ld %lf %lf %lf %lf\n", tolower(dt_type),
            (dim_t)m, (dim_t)incx, (dim_t)incy, c_real, c_imag, s_real, s_imag);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_rotg_sizes(int8 loglevel,
                             char dt_type,
                             const void *a,
                             const void *b,
                             const void *c,
                             const void *s,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} {a, b, c, s}

    double a_real = 0.0;
    double a_imag = 0.0;
    double b_real = 0.0;
    double b_imag = 0.0;
    double c_real = 0.0;
    double c_imag = 0.0;
    double s_real = 0.0;
    double s_imag = 0.0;

    DTL_get_complex_parts(dt_type, a, &a_real, &a_imag);
    DTL_get_complex_parts(dt_type, b, &b_real, &b_imag);
    DTL_get_complex_parts(dt_type, c, &c_real, &c_imag);
    DTL_get_complex_parts(dt_type, s, &s_real, &s_imag);

    sprintf(buffer, "%c %lf %lf %lf %lf %lf %lf %lf %lf\n", tolower(dt_type),
            a_real, a_imag, b_real, b_imag, c_real, c_imag, s_real, s_imag);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_rotm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int incx,
                             const f77_int incy,
                             const void *param1,
                             const void *param2,
                             const void *param3,
                             const void *param4,
                             const void *param5,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D,C, Z} {m, incx, incy, param[5]}

    double dflag_real = 0.0;
    double dh11_real  = 0.0;
    double dh21_real  = 0.0;
    double dh12_real  = 0.0;
    double dh22_real  = 0.0;
    double tmp_imag   = 0.0;

    DTL_get_complex_parts(dt_type, param1, &dflag_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, param2, &dh11_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, param3, &dh21_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, param4, &dh12_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, param5, &dh22_real, &tmp_imag);

    // No complex variant of this API, so don't print complex parts
    sprintf(buffer, "%c %ld %ld %ld %lf %lf %lf %lf %lf\n", tolower(dt_type),
            (dim_t)m, (dim_t)incx, (dim_t)incy, dflag_real, dh11_real, dh21_real, dh12_real, dh22_real);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_rotmg_sizes(int8 loglevel,
                              char dt_type,
                              const void* dd1,
                              const void* dd2,
                              const void* dx1,
                              const void* dy1,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];
    // {S, D,C, Z} {dd1, dd2, dx1, dy1}

    double dd1_real = 0.0;
    double dd2_real = 0.0;
    double dx1_real = 0.0;
    double dy1_real = 0.0;
    double tmp_imag = 0.0;

    DTL_get_complex_parts(dt_type, dd1, &dd1_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, dd2, &dd2_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, dx1, &dx1_real, &tmp_imag);
    DTL_get_complex_parts(dt_type, dy1, &dy1_real, &tmp_imag);

    // No complex variant of this API, so don't print complex parts
    sprintf(buffer, "%c %lf %lf %lf %lf\n", tolower(dt_type),
            dd1_real, dd2_real, dx1_real, dy1_real);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-1 Logging

void AOCL_DTL_log_amax_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, "%c %ld %ld\n", tolower(dt_type), (dim_t)n, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_asum_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, "%c %ld %ld\n", tolower(dt_type), (dim_t)n, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_axpby_sizes(int8 loglevel,
                              char dt_type,
                              const f77_int n,
                              const void *alpha,
                              const f77_int incx,
                              const void *beta,
                              const f77_int incy,
                              const char *filename,
                              const char *function_name,
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // {S, D, C, Z} {n, alpha_real, alpha_imag, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %ld %lf %lf %ld %lf %lf %ld\n", tolower(dt_type),
            (dim_t)n, alpha_real, alpha_imag, (dim_t)incx,
            beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_axpy_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const void *alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D, C, Z} {n, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %ld %lf %lf %ld %ld\n", tolower(dt_type),
            (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_copy_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx, incy}
    sprintf(buffer, "%c %ld %ld %ld\n", tolower(dt_type), (dim_t)n, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_dotv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char conjx,
                             const f77_int n,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];

    // { conjx, n, incx, incy}
    sprintf(buffer, "%c %c %ld %ld %ld\n",  tolower(dt_type), conjx, (dim_t)n, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_nrm2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, "%c %ld %ld", tolower(dt_type), (dim_t)n, (dim_t)incx);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_nrm2_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n)
{
    char buffer[256];

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

    double flops = 2.0 * n;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 2.0 * flops;
    }

    if (execution_time != 0.0)
        sprintf(buffer, " nt=%ld %.3f ms %0.3f GFLOPS",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0,
                flops/(execution_time * 1e3));
    else
        sprintf(buffer, " nt=%ld %.3f ms",
                AOCL_get_requested_threads_count(),
                execution_time/1000.0);

    DTL_Trace(loglevel, TRACE_TYPE_RAW, NULL, NULL, 0, buffer);
}

void AOCL_DTL_log_scal_sizes(int8 loglevel,
                             char dt_type,
                             const void *alpha,
                             const f77_int n,
                             const f77_int incx,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);

    // {S, D, C, Z} { alpha, n, incx}
    sprintf(buffer, "%c %lf %lf %ld %ld\n", tolower(dt_type),
            alpha_real, alpha_imag, (dim_t)n, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_swap_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const f77_int incy,
                             const char *filename,
                             const char *function_name,
                             int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx, incy}
    sprintf(buffer, "%c %ld %ld %ld\n", tolower(dt_type),
            (dim_t)n, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

#endif
