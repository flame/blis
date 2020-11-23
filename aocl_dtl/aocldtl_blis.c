/*===================================================================
 * File Name :  aocldtl_blis.c
 *
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/


#include "blis.h"

#if AOCL_DTL_LOG_ENABLE

// Level-3

void AOCL_DTL_log_gemm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_char transb,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const void* alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void* beta,
                             const f77_int ldc,
                             const char* filename,
                             const char* function_name,
                             int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if( dt_type == 'S' || dt_type == 's' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if( dt_type == 'D' || dt_type == 'd' )
    {
        alpha_real = *(double*)alpha;
        alpha_imag = 0.0;
        beta_real = *(double*) beta;
        beta_imag = 0.0;
    }
    else if( dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if( dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }

    //{S, D, C, Z} m, n, k, lda, ldb, ldc, transa, transb, alpha_real, alpha_imag, beta_real, beta_imag
    sprintf(buffer, "%c %ld %ld %ld %ld %ld %ld %c %c %lf %lf %lf %lf",
                    dt_type,
                    (dim_t)m, (dim_t)n, (dim_t)k,
                    (dim_t)lda, (dim_t)ldb, (dim_t)ldc,
                    transa, transb,
                    alpha_real, alpha_imag, beta_real, beta_imag
                    );

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
                 const void* alpha,
                 f77_int lda,
                 f77_int ldb,
                 const char* filename,
                 const char* function_name,
                 int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if( dt_type == 'S' || dt_type == 's' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if( dt_type == 'D' || dt_type == 'd' )
    {
        alpha_real = *(double*)alpha;
        alpha_imag = 0.0;
    }
    else if( dt_type == 'C' || dt_type == 'c' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if( dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }

    //{S, D, C, Z} side, uplo, transa, diaga, m, n, lda, ldb, alpha_real, alpha_imag
    sprintf(buffer, "%c %c %c %c %c %ld %ld %ld %ld %lf %lf",dt_type,
                    side, uploa, transa, diaga,
                    (dim_t)m, (dim_t)n, (dim_t)lda, (dim_t)ldb,
                    alpha_real, alpha_imag
                    );

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
                 int line)
{
    char buffer[256];

    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if( dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if( dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real = *(double*) beta;
        beta_imag = 0.0;
    }
    else if( dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if( dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }

    // {S,D,C,Z} {triangC : l or u} {n k lda ldb ldc transa transb alpha_real alpha_imaginary
    // beta_real, beta_imaginary}
    sprintf(buffer, " %c %c %ld %ld %lu %lu %lu %c %c %lf %lf %lf %lf",
            dt_type, uplo, (dim_t)n, (dim_t)k,
                (dim_t)lda, (dim_t)ldb, (dim_t)ldc,
                transa, transb,
                alpha_real, alpha_imag,
                beta_real, beta_imag);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
                             int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }

    // {C, Z} { side, uploa, m, n, alpha_real, alpha_imag, lda, incx, beta_real, beta_imag, incy}

    sprintf(buffer, "%c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, side, uploa, (dim_t)m, (dim_t)n, alpha_real, alpha_imag,
            (dim_t)lda, (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-3
void AOCL_DTL_log_herk_sizes( int8 loglevel,
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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;
    if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (double)(((scomplex*)alpha)->real);
        alpha_imag = (double)(((scomplex*)alpha)->imag);
        beta_real = (double)(((scomplex*)beta)->real);
        beta_imag = (double)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = (double)((dcomplex*)alpha)->real;
        alpha_imag = (double)((dcomplex*)alpha)->imag;
        beta_real = (double)((dcomplex*)beta)->real;
        beta_imag = (double)((dcomplex*)beta)->imag;
    }
    // {C, Z} {uploc, transa, m, k, alpha_real, alpha_imag, lda, beta_real, beta_imag, ldc}
    sprintf(buffer, " %c %c %c %ld %ld %lf %lf %ld %lf %lf %ld",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k,  alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;
    if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (double)(((scomplex*)alpha)->real);
        alpha_imag = (double)(((scomplex*)alpha)->imag);
        beta_real = (double)(((scomplex*)beta)->real);
        beta_imag = (double)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = (double)((dcomplex*)alpha)->real;
        alpha_imag = (double)((dcomplex*)alpha)->imag;
        beta_real = (double)((dcomplex*)beta)->real;
        beta_imag = (double)((dcomplex*)beta)->imag;
    }
    // {C, Z} { uploc, transa, m, k, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, " %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k,  alpha_real, alpha_imag, (dim_t)lda,  (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_symm_sizes( int8 loglevel,
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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real = *(double*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }

    // {S, D, C, Z} { side, uploa, m, n, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, " %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, side, uploa, (dim_t)m, (dim_t)n,  alpha_real, alpha_imag, (dim_t)lda,  (dim_t)ldb, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

// Level-2
void AOCL_DTL_log_symv_sizes( int8 loglevel,
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
                              int line)
{
    char buffer[256];
    double alpha_d = 0.0;
    double beta_d = 0.0;
    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_d = *(float*)alpha;
        beta_d = *(float*)beta;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_d = *(double*) alpha;
        beta_d = *(double*) beta;
    }

    // {S, D} { uploa, m, alpha_d, lda, incx, beta_d, incy}
    sprintf(buffer, " %c %c %ld %lf %ld %ld %lf %ld",
            dt_type, uploa, (dim_t)m, alpha_d, (dim_t)lda,  (dim_t)incx, beta_d, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;
    f77_char transaUpdate = transa;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real = *(double*) beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }
    /* The following convention is followed to print trans character
     * BLIS_NO_TRANSPOSE  = 'n';
     * BLIS_TRANSPOSE     = 't';
     * BLIS_CONJ_NO_TRANS = 'c';
     * BLIS_CONJ_TRANS    = 'h';
    */
    if( transa == BLIS_NO_TRANSPOSE )
    {
        transaUpdate = 'n';
    }
    else if( transa == BLIS_TRANSPOSE )
    {
        transaUpdate = 't';
    }
    else if( transa == BLIS_CONJ_NO_TRANSPOSE )
    {
        transaUpdate = 'c';
    }
    else if( transa == BLIS_CONJ_NO_TRANSPOSE )
    {
     transaUpdate = 'h';
    }

    // {S, D,C, Z} { transa, m, n, alpha, lda, incx, beta, incy}
    sprintf(buffer, " %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, transaUpdate, (dim_t)m, (dim_t)n,  alpha_real, alpha_imag,
            (dim_t)lda,  (dim_t)incx, beta_real, beta_imag, (dim_t)incy);


    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

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
                          )
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }

    sprintf(buffer, "%c %ld %ld %lf %lf %ld %ld %ld", dt_type, (dim_t)m, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda );

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_her_sizes( int8 loglevel,
                              char dt_type,
                              const f77_char uploa,
                              const f77_int  m,
                              const void* alpha,
                              const f77_int  incx,
                              const f77_int lda,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (double)(((scomplex*)alpha)->real);
        alpha_imag = (double)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = (double)((dcomplex*)alpha)->real;
        alpha_imag = (double)((dcomplex*)alpha)->imag;
    }
    // {C, Z} {uploa, m alpha_real, alpha_imag incx lda}
    sprintf(buffer, " %c %c %ld %lf %lf %ld %ld",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_dotv_sizes( int8 loglevel,
                              char dt_type,
                              char transa,
                              const f77_int  n,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];

    // { n, incx, incy}
    sprintf(buffer, " %c %c %ld %ld %ld", dt_type, transa, (dim_t)n, (dim_t)incx, (dim_t)incy);


    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real = *(double*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }
    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, lda, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)lda, (dim_t)incx, beta_real, beta_imag, (dim_t)incy);


    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}


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
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }

    // {S, D, C, Z} {uploa, m, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %c %ld %lf %lf %ld %ld",
                    dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

// Level-1

void AOCL_DTL_log_amax_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int incx,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, "%c %ld %ld", dt_type, (dim_t)n, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_asum_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int incx,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, "%c %ld %ld", dt_type, (dim_t)n, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_axpby_sizes ( int8 loglevel,
                               char dt_type,
                               const f77_int  n,
                               const void* alpha,
                               const f77_int incx,
                               const void* beta,
                               const f77_int incy,
                               const char* filename,
                               const char* function_name,
                               int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real = *(float*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real = *(double*)beta;
        beta_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }

    // {S, D, C, Z} {n, alpha_real, alpha_imag, incx, beta_real, beta_imag, incy}
    sprintf(buffer, "%c %ld %lf %lf %ld %lf %lf %ld",
                    dt_type, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx,
                    beta_real, beta_imag, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_axpy_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const void* alpha,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }

    // {S, D, C, Z} {n, alpha_real, alpha_imag, incx, incy}
    sprintf(buffer, "%c %ld %lf %lf %ld %ld",
                    dt_type, (dim_t)n, alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_copy_sizes( int8 loglevel,
                              char dt_type,
                              const f77_int n,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line
                            )
{
    char buffer[256];
    // {S, D, C, Z} {n, incx, incy}
    sprintf(buffer, "%c %ld %ld %ld", dt_type, (dim_t)n, (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}


void AOCL_DTL_log_scal_sizes( int8 loglevel,
                              char dt_type,
                              const double alpha,
                              const f77_int  n,
                              const f77_int  incx,
                              const char* filename,
                              const char* function_name,
                              int line)
{
  char buffer[256];
    // {S, D, C, Z} { alpha, n, incx}
    sprintf(buffer, " %c %lf %ld %ld",
            dt_type, alpha, (dim_t)n,  (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_swap_sizes( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int  incx,
                              const f77_int  incy,
                              const char* filename,
                              const char* function_name,
                              int line)
{
  char buffer[256];
    // {S, D, C, Z} {n, incx, incy}
    sprintf(buffer, " %c %ld %ld %ld",
            dt_type, (dim_t)n,  (dim_t)incx, (dim_t)incy);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

void AOCL_DTL_log_nrm2_sizes( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int  incx,
                              const char* filename,
                              const char* function_name,
                              int line)
{
  char buffer[256];
    // {S, D, C, Z} {n, incx}
    sprintf(buffer, " %c %ld %ld",
            dt_type, (dim_t)n,  (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}

//Level-2
void AOCL_DTL_log_syr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int  incx,
                             const f77_int  incy,
                             const f77_int  lda,
                             const char*  filename,
                             const char*  function_name,
                             int  line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }
    // { uploa, m, alpha_real, alpha_imag, incx, incy, lda}
    sprintf(buffer, " %c %c %ld %lf %lf %ld %ld %ld",
            dt_type, uploa, (dim_t)m,  alpha_real, alpha_imag, (dim_t)incx, (dim_t)incy, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_syr2k_sizes(int8  loglevel,
                             char dt_type,
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
                             int  line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real  = *(float*)beta;
        beta_imag  = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real  = *(double*) beta;
        beta_imag  = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real  = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real  = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }
    // { uploc, transa, m, k, alpha_real, alpha_imag, lda, ldb, beta_real, beta_imag, ldc}
    sprintf(buffer, " %c %c %c %ld %ld %lf %lf %ld %ld %lf %lf %ld",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k,  alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb, beta_real, beta_imag ,(dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_syr_sizes(int8  loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int  incx,
                             const f77_int  lda,
                             const char*    filename,
                             const char*    function_name,
                             int  line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }
    // {S, D,C, Z} { uploa, m, alpha_real, alpha_imag, incx, lda}
    sprintf(buffer, " %c %c %ld %lf %lf %ld %ld",
            dt_type, uploa, (dim_t)m, alpha_real, alpha_imag, (dim_t)incx, (dim_t)lda);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
                             int  line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;
    double beta_real = 0.0;
    double beta_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
        beta_real  = *(float*)beta;
        beta_imag  = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
        beta_real  = *(double*) beta;
        beta_imag  = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
        beta_real  = (float)(((scomplex*)beta)->real);
        beta_imag = (float)(((scomplex*)beta)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
        beta_real  = ((dcomplex*)beta)->real;
        beta_imag = ((dcomplex*)beta)->imag;
    }
    // {S, D,C, Z} { uploc, transa, m, k, alpha_real, alpha_imag, lda, beta_real, beta_imag, ldc}
    sprintf(buffer, " %c %c %c %ld %ld %lf %lf %ld %lf %lf %ld",
            dt_type, uploc, transa, (dim_t)m, (dim_t)k,  alpha_real, alpha_imag, (dim_t)lda, beta_real, beta_imag, (dim_t)ldc);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

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
                             int  line)
{
    char buffer[256];
    double alpha_real = 0.0;
    double alpha_imag = 0.0;

    if(dt_type == 's' || dt_type == 'S' )
    {
        alpha_real = *(float*)alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'd' || dt_type == 'D' )
    {
        alpha_real = *(double*) alpha;
        alpha_imag = 0.0;
    }
    else if(dt_type == 'c' || dt_type == 'C' )
    {
        alpha_real = (float)(((scomplex*)alpha)->real);
        alpha_imag = (float)(((scomplex*)alpha)->imag);
    }
    else if(dt_type == 'z' || dt_type == 'Z' )
    {
        alpha_real = ((dcomplex*)alpha)->real;
        alpha_imag = ((dcomplex*)alpha)->imag;
    }
    // {S, D,C, Z} { side, uploa, transa, diaga, m, n, alpha_real, alpha_imag, lda, ldb}
    sprintf(buffer, " %c %c %c %c %c %ld %ld %lf %lf %ld %ld",
            dt_type, side, uploa, transa, diaga, (dim_t)m, (dim_t)n,  alpha_real, alpha_imag, (dim_t)lda, (dim_t)ldb);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int  m,
                             const f77_int  lda,
                             const f77_int  incx,
                             const char*    filename,
                             const char*    function_name,
                             int  line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, lda, incx}
    sprintf(buffer, " %c %c %c %c %ld %ld %ld",
            dt_type, uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int  m,
                             const f77_int  lda,
                             const f77_int  incx,
                             const char*    filename,
                             const char*    function_name,
                             int  line)
{
    char buffer[256];
    // {S, D,C, Z} { side, uploa, transa, diaga, m, lda, incx}
    sprintf(buffer, " %c %c %c %c %ld %ld %ld",
            dt_type, uploa, transa, diaga, (dim_t)m, (dim_t)lda, (dim_t)incx);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}
#endif
