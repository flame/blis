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

void AOCL_DTL_log_gemm_sizes(int8 loglevel,
                             obj_t* alpha,
                             obj_t* a,
                             obj_t* b,
                             obj_t* beta,
                             obj_t* c,
                             const char* filename,
                             const char* function_name,
                             int line)
{
    char buffer[256];
    gint_t m = bli_obj_length( c );
    gint_t n = bli_obj_width( c );
    gint_t k = bli_obj_length( b );
    guint_t csa = bli_obj_col_stride( a );
    guint_t csb = bli_obj_col_stride( b );
    guint_t csc = bli_obj_col_stride( c );
    guint_t rsa = bli_obj_row_stride( a );
    guint_t rsb = bli_obj_row_stride( b );
    guint_t rsc = bli_obj_row_stride( c );
    const num_t dt_exec = bli_obj_dt( c );
    char transa, transb;
    double alpha_r, alpha_i, beta_r, beta_i;

    /* The following convention is followed to print trans character
     * BLIS_NO_TRANSPOSE  = 'n';
     * BLIS_TRANSPOSE     = 't';
     * BLIS_CONJ_NO_TRANS = 'c';
     * BLIS_CONJ_TRANS    = 'h';
     */

    if(bli_obj_has_trans(a))
    {
        if(bli_obj_has_conj(a))
            transa = 'h';
        else
            transa = 't';
    }
    else
    {
        if(bli_obj_has_conj(a))
            transa = 'c';
        else
            transa = 'n';
    }

    if(bli_obj_has_trans(b))
    {
        if(bli_obj_has_conj(b))
            transb = 'h';
        else
            transb = 't';
    }
    else
    {
        if(bli_obj_has_conj(b))
            transb = 'c';
        else
            transb = 'n';
    }

    char dt_type = 'D'; // to indicate data-type of this GEMM operation

    if ( bli_is_float(dt_exec) )
      {
        float* alpha_cast = (float*)bli_obj_buffer_for_1x1( dt_exec, alpha );
        float* beta_cast  = (float*)bli_obj_buffer_for_1x1( dt_exec, beta );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;
        beta_r  = *beta_cast;
        beta_i  = 0.0;

        dt_type = 'S';
      }
    else if ( bli_is_double(dt_exec) )
      {
        double* alpha_cast = (double*)bli_obj_buffer_for_1x1( dt_exec, alpha );
        double* beta_cast  = (double*)bli_obj_buffer_for_1x1( dt_exec, beta );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;
        beta_r  = *beta_cast;
        beta_i  = 0.0;
        dt_type = 'D';
      }
    else if(bli_is_scomplex(dt_exec))
      {
        scomplex* alpha_cast = (scomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        scomplex* beta_cast  = (scomplex*)bli_obj_buffer_for_1x1(dt_exec, beta);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);
        beta_r  = (double)(beta_cast->real);
        beta_i  = (double)(beta_cast-> imag);
        dt_type = 'C';
      }
    else
      {
        dcomplex* alpha_cast = (dcomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        dcomplex* beta_cast  = (dcomplex*)bli_obj_buffer_for_1x1(dt_exec, beta);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);
        beta_r  = (double)(beta_cast->real);
        beta_i  = (double)(beta_cast-> imag);
        dt_type = 'Z';
      }


// {S,D,C,Z} {m n k cs_a cs_b cs_c rs_a rs_b rs_c transa transb alpha_real alpha_imaginary beta_real beta_imaginary}

    sprintf(buffer, " %c %ld %ld %ld %lu %lu %lu %lu %lu %lu %c %c %lf %lf %lf %lf",
            dt_type, m, n, k,
                 csa, csb, csc,
                 rsa, rsb, rsc,
                 transa, transb,
                 alpha_r, alpha_i,
                 beta_r, beta_i);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_trsm_sizes(int8 loglevel,
                 side_t side,
                 obj_t* alpha,
                 obj_t* a,
                 obj_t* b,
                 const char* filename,
                 const char* function_name,
                 int line)
{
    char buffer[256];
    gint_t m = bli_obj_length(b);
    gint_t n = bli_obj_width(b);
    guint_t csa = bli_obj_col_stride(a);
    guint_t csb = bli_obj_col_stride(b);
    guint_t rsa = bli_obj_row_stride(a);
    guint_t rsb = bli_obj_row_stride(b);
    const num_t dt_exec = bli_obj_dt(b);
    char transa;
    char diaga;
    double alpha_r, alpha_i;

    /* The following convention is followed to print trans character
     * BLIS_NO_TRANSPOSE  = 'n';
     * BLIS_TRANSPOSE     = 't';
     * BLIS_CONJ_NO_TRANS = 'c';
     * BLIS_CONJ_TRANS    = 'h';
     */

    if(bli_obj_has_trans(a))
    {
        if(bli_obj_has_conj(a))
            transa = 'h';
        else
            transa = 't';
    }
    else
    {
        if(bli_obj_has_conj(a))
            transa = 'c';
        else
            transa = 'n';
    }

    if( bli_obj_has_unit_diag(a) ) diaga = 'u';
    else                 diaga = 'n';


    char dt_type = 'D'; // to indicate data-type of this TRSM operation

    if ( bli_is_float(dt_exec) )
      {
        float* alpha_cast = (float*)bli_obj_buffer_for_1x1( dt_exec, alpha );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;

        dt_type = 'S';
      }
    else if ( bli_is_double(dt_exec) )
      {
        double* alpha_cast = (double*)bli_obj_buffer_for_1x1( dt_exec, alpha );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;

        dt_type = 'D';
      }
    else if( bli_is_scomplex(dt_exec) )
      {
        scomplex* alpha_cast = (scomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);

        dt_type = 'C';
      }
    else
      {
        dcomplex* alpha_cast = (dcomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);

        dt_type = 'Z';
      }

// {S,D,C,Z} {side : L or R} {Triangular: U or L} {m n cs_a cs_b rs_a rs_b transa diaga{unit u, non-unit - u}
    // alpha_real alpha_imaginary }
    sprintf(buffer, " %c %c %c %ld %ld %lu %lu %lu %lu %c %c %lf %lf",
            dt_type, (bli_is_right(side) ? 'R' : 'L'),
            ( bli_obj_is_upper(a) ? 'U' : bli_obj_is_lower(a) ? 'L' : 'N'),
            m, n,
            csa, csb,
            rsa, rsb,
            transa,
            diaga,
            alpha_r, alpha_i);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}

void AOCL_DTL_log_gemmt_sizes(int8 loglevel,
                 obj_t* alpha,
                 obj_t* a,
                 obj_t* b,
                 obj_t* beta,
                 obj_t* c,
                 const char* filename,
                 const char* function_name,
                 int line)
{
    char buffer[256];
    gint_t n = bli_obj_length(c);
    gint_t k = bli_obj_width_after_trans(a);
    guint_t csa = bli_obj_col_stride(a);
    guint_t csb = bli_obj_col_stride(b);
    guint_t csc = bli_obj_col_stride(c);
    guint_t rsa = bli_obj_row_stride(a);
    guint_t rsb = bli_obj_row_stride(b);
    guint_t rsc = bli_obj_row_stride(c);
    const num_t dt_exec = bli_obj_dt(c);
    char transa, transb;
    double alpha_r, alpha_i, beta_r, beta_i;

    /* The following convention is followed to print trans character
     * BLIS_NO_TRANSPOSE  = 'n';
     * BLIS_TRANSPOSE     = 't';
     * BLIS_CONJ_NO_TRANS = 'c';
     * BLIS_CONJ_TRANS    = 'h';
     */
    if(bli_obj_has_trans(a))
    {
        if(bli_obj_has_conj(a))
            transa = 'h';
        else
            transa = 't';
    }
    else
    {
        if(bli_obj_has_conj(a))
            transa = 'c';
        else
            transa = 'n';
    }

    if(bli_obj_has_trans(b))
    {
        if(bli_obj_has_conj(b))
            transb = 'h';
        else
            transb = 't';
    }
    else
    {
        if(bli_obj_has_conj(b))
            transb = 'c';
        else
            transb = 'n';
    }

    char triangC;
    char dt_type = 'D'; // to indicate data-type of this GEMMT operation

    if(bli_obj_is_lower(c)) triangC = 'l';
    else                    triangC = 'u';

    if ( bli_is_float(dt_exec) )
      {
        float* alpha_cast = (float*)bli_obj_buffer_for_1x1( dt_exec, alpha );
        float* beta_cast  = (float*)bli_obj_buffer_for_1x1( dt_exec, beta );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;
        beta_r  = *beta_cast;
        beta_i  = 0.0;

        dt_type = 'S';
      }
    else if ( bli_is_double(dt_exec) )
      {
        double* alpha_cast = (double*)bli_obj_buffer_for_1x1( dt_exec, alpha );
        double* beta_cast  = (double*)bli_obj_buffer_for_1x1( dt_exec, beta );

        alpha_r = *alpha_cast;
        alpha_i = 0.0;
        beta_r  = *beta_cast;
        beta_i  = 0.0;
        dt_type = 'D';
      }
    else if(bli_is_scomplex(dt_exec))
      {
        scomplex* alpha_cast = (scomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        scomplex* beta_cast  = (scomplex*)bli_obj_buffer_for_1x1(dt_exec, beta);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);
        beta_r  = (double)(beta_cast->real);
        beta_i  = (double)(beta_cast-> imag);

        dt_type = 'C';
      }
    else
      {
        dcomplex* alpha_cast = (dcomplex*)bli_obj_buffer_for_1x1(dt_exec, alpha);
        dcomplex* beta_cast  = (dcomplex*)bli_obj_buffer_for_1x1(dt_exec, beta);
        alpha_r = (double)(alpha_cast->real);
        alpha_i = (double)(alpha_cast->imag);
        beta_r  = (double)(beta_cast->real);
        beta_i  = (double)(beta_cast-> imag);

        dt_type = 'Z';
      }

    // {S,D,C,Z} {triangC : l or u} {n k cs_a cs_b cs_c rs_a rs_b rs_c transa transb alpha_real alpha_imaginary
    // beta_real, beta_imaginary}
    sprintf(buffer, " %c %c %ld %ld %lu %lu %lu %lu %lu %lu %c %c %lf %lf %lf %lf",
            dt_type, triangC, n, k,
                csa, csb, csc,
                rsa, rsb, rsc,
                transa, transb,
                alpha_r, alpha_i,
                beta_r, beta_i);

    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);
}


// Level-2

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
                              int line)
{
    char buffer[256];
    // {S, D,C, Z} { transa, m, n, alpha, lda, incx, beta, incy}
    sprintf(buffer, " %c %c %ld %ld %lf %lu %lu %lf %lu",
            dt_type, transa, (dim_t)m, (dim_t)n,  alpha, (dim_t)lda,  (dim_t)incx, beta, (dim_t)incy);


    DTL_Trace(loglevel, TRACE_TYPE_LOG, function_name, function_name, line, buffer);

}


#endif
