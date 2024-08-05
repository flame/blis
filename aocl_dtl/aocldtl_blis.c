/*===================================================================
 * File Name :  aocldtl_blis.c
 *
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/

#include "blis.h"

dim_t AOCL_get_requested_threads_count(void)
{
    return bli_thread_get_num_threads();
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

// Level-3


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

    AOCL_DTL_START_PERF_TIMER();

    DTL_get_complex_parts(dt_type, alpha, &alpha_real, &alpha_imag);
    DTL_get_complex_parts(dt_type, beta, &beta_real, &beta_imag);

    // Ordering as per cblas/blas interfaces
    // {S, D, C, Z} transa, transb, m, n, k, alpha_real, alpha_imag,
    //              lda, ldb, beta_real, beta_imag, ldc
    sprintf(buffer, "%c %c %c %ld %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            toupper(dt_type),
            toupper(transa), toupper(transb),
            (dim_t)m, (dim_t)n, (dim_t)k,
            alpha_real, alpha_imag,
            (inc_t)lda, (inc_t)ldb,
            beta_real, beta_imag,
            (inc_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_gemm_stats(int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k)
{
    char buffer[256];

    double flops = 2.0 * m * n * k;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 4.0 * flops;
    }

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

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

void AOCL_DTL_log_gemmt_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int k)
{
    char buffer[256];

    double flops = n * n * k;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 4.0 * flops;
    }

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

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
    sprintf(buffer, "%c %c %c %c %c %ld %ld %ld %ld %lf %lf", dt_type,
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

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

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
            dt_type, uplo, (dim_t)n, (dim_t)k,
            (dim_t)lda, (dim_t)ldb, (dim_t)ldc,
            transa, transb,
            alpha_real, alpha_imag,
            beta_real, beta_imag);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
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
            dt_type, side, uploa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag,
            (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-3
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
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %lf %lf %ld\n",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

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
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

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
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
            dt_type, side, uploa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-2
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
    sprintf(buffer, "%c %c %ld %lf %ld %ld %lf %ld\n",
            dt_type, uploa, (dim_t)m, alpha_d, (dim_t)lda, (dim_t)incx, beta_d, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
    sprintf(buffer, "%c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
            dt_type, transa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag,
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

    sprintf(buffer, "%c %ld %ld %lf %lf %ld %ld %ld\n", dt_type, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda);

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
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

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
    sprintf(buffer, "%c %c %ld %ld %ld\n", dt_type, conjx, (dim_t)n, (dim_t)incx, (dim_t)incy);

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
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld %lf %lf %ld\n",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);

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
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-1

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
    sprintf(buffer, "%c %ld %ld\n", dt_type, (dim_t)n, (dim_t)incx);

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
    sprintf(buffer, "%c %ld %ld\n", dt_type, (dim_t)n, (dim_t)incx);

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
    sprintf(buffer, "%c %ld %lf %lf %ld %lf %lf %ld\n",
            dt_type, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx,
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
    sprintf(buffer, "%c %ld %lf %lf %ld %ld\n",
            dt_type, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

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
    sprintf(buffer, "%c %ld %ld %ld\n", dt_type, (dim_t)n, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
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
    sprintf(buffer, "%c %lf %lf %ld %ld\n",
            dt_type, alpha_real, alpha_imag, (dim_t)n, (dim_t)incx);

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
    sprintf(buffer, "%c %ld %ld %ld\n",
            dt_type, (dim_t)n, (dim_t)incx, (dim_t)incy);

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
    sprintf(buffer, "%c %ld %ld",
            dt_type, (dim_t)n, (dim_t)incx);

    AOCL_DTL_START_PERF_TIMER();
    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_nrm2_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n)
{
    char buffer[256];

    double flops = 2.0 * n;
    if (dt_type == 'c' || dt_type == 'C' || dt_type == 'z' || dt_type == 'Z')
    {
        flops = 2.0 * flops;
    }

    // Execution time is in micro seconds.
    Double execution_time = AOCL_DTL_get_time_spent();

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

//Level-2
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
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld %ld\n",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda);

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
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld\n",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

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
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld\n",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

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
    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %lf %lf %ld\n",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k, alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

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
    sprintf(buffer, "%c %c %c %c %c %ld %ld %lf %lf %ld %ld\n",
            dt_type, side, uploa, transa, diaga, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb);

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
    sprintf(buffer, "%c %c %c %c %ld %ld %ld\n",
            dt_type, uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

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
    sprintf(buffer, "%c %c %c %c %ld %ld %ld\n",
            dt_type, uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}
#endif
