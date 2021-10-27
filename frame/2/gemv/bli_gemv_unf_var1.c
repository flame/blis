/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 21, Advanced Micro Devices, Inc. All rights reserved.

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
#define BLIS_DGEMV_VAR1_FUSE 8

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
\
    if(cntx == NULL) cntx = bli_gks_query_cntx(); \
\
    const num_t dt = PASTEMAC(ch,type); \
\
    ctype*  A1; \
    ctype*  x1; \
    ctype*  y1; \
    dim_t   i; \
    dim_t   b_fuse, f; \
    dim_t   n_elem, n_iter; \
    inc_t   rs_at, cs_at; \
    conj_t  conja; \
\
    bli_set_dims_incs_with_trans( transa, \
                                  m, n, rs_a, cs_a, \
                                  &n_iter, &n_elem, &rs_at, &cs_at ); \
\
    conja = bli_extract_conj( transa ); \
\
    PASTECH(ch,dotxf_ker_ft) kfp_df; \
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
    for ( i = 0; i < n_iter; i += f ) \
    { \
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
        A1 = a + (i  )*rs_at + (0  )*cs_at; \
        x1 = x + (0  )*incy; \
        y1 = y + (i  )*incy; \
\
        /* y1 = beta * y1 + alpha * A1 * x; */ \
        kfp_df \
        ( \
          conja, \
          conjx, \
          n_elem, \
          f, \
          alpha, \
          A1,   cs_at, rs_at, \
          x1,   incx, \
          beta, \
          y1,   incy, \
          cntx  \
        ); \
\
    } \
}

#ifdef BLIS_CONFIG_EPYC
void bli_dgemv_unf_var1
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double*  alpha,
       double*  a, inc_t rs_a, inc_t cs_a,
       double*  x, inc_t incx,
       double*  beta,
       double*  y, inc_t incy,
       cntx_t* cntx
     )
{

    double*  A1;
    double*  y1;
    dim_t   i;
    dim_t   f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;
    //memory pool declarations for packing vector X.
    mem_t   mem_bufX;
    rntm_t  rntm;
    double  *x_buf = x;
    inc_t   buf_incx = incx;

    bli_init_once();

    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_iter, &n_elem, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
    // This function is invoked on all architectures including ‘generic’.
    // Invoke architecture specific kernels only if we are sure that we are running on zen,
    // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
    arch_t id = bli_arch_query_id();
    bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

    if (bamdzen == 0)
    {
        if ( cntx == NULL ) cntx = bli_gks_query_cntx();
        const num_t dt = PASTEMAC(d,type);
        double*  x1;
        double*  y1;
        PASTECH(d,dotxf_ker_ft) kfp_df;
        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
        dim_t b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (i  )*rs_at + (0  )*cs_at;
            x1 = x + (0  )*incy;
            y1 = y + (i  )*incy;

            /* y1 = beta * y1 + alpha * A1 * x; */
            kfp_df
            (
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1,   cs_at, rs_at,
            x1,   incx,
            beta,
            y1,   incy,
            cntx
            );

        }

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
      return;
    }

    if (incx > 1)
    {
        /*
          Initialize mem pool buffer to NULL and size to 0
          "buf" and "size" fields are assigned once memory
          is allocated from the pool in bli_membrk_acquire_m().
          This will ensure bli_mem_is_alloc() will be passed on
          an allocated memory if created or a NULL .
        */
        mem_bufX.pblk.buf = NULL;   mem_bufX.pblk.block_size = 0;
        mem_bufX.buf_type = 0;      mem_bufX.size = 0;
        mem_bufX.pool = NULL;

        /* In order to get the buffer from pool via rntm access to memory broker
        is needed.Following are initializations for rntm */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_membrk_rntm_set_membrk( &rntm );

        //calculate the size required for n_elem double elements in vector X.
        size_t buffer_size = n_elem * sizeof(double);

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_unf_var1(): get mem pool block\n" );
        #endif

        /*acquire a Buffer(n_elem*size(double)) from the memory broker
        and save the associated mem_t entry to mem_bufX.*/
        bli_membrk_acquire_m(&rntm,
                                buffer_size,
                                BLIS_BUFFER_FOR_B_PANEL,
                                &mem_bufX);

        /*Continue packing X if buffer memory is allocated*/
        if ((bli_mem_is_alloc( &mem_bufX )))
        {
            x_buf = bli_mem_buffer(&mem_bufX);

            //pack X vector with non-unit stride to a temp buffer x_buf with unit stride
            for(dim_t x_index = 0 ; x_index < n_elem ; x_index++)
            {
                *(x_buf + x_index) =  *(x + (x_index * incx)) ;
            }
            // stride of vector x_buf =1
            buf_incx = 1;
        }
    }

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, BLIS_DGEMV_VAR1_FUSE );

        A1 = a + (i  )*rs_at + (0  )*cs_at;
        y1 = y + (i  )*incy;

        /* y1 = beta * y1 + alpha * A1 * x; */
        bli_ddotxf_zen_int_8
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1,   cs_at, rs_at,
          x_buf,   buf_incx,
          beta,
          y1,   incy,
          cntx
        );

    }
    if ((incx > 1) && bli_mem_is_alloc( &mem_bufX ))
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_unf_var1(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool
        bli_membrk_release(&rntm , &mem_bufX);
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

void bli_sgemv_unf_var1
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       float*  alpha,
       float*  a, inc_t rs_a, inc_t cs_a,
       float*  x, inc_t incx,
       float*  beta,
       float*  y, inc_t incy,
       cntx_t* cntx
     )
{

    float*  A1;
    float*  x1;
    float*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    bli_init_once();

    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_iter, &n_elem, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
    // This function is invoked on all architectures including ‘generic’.
    // Invoke architecture specific kernels only if we are sure that we are running on zen,
    // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
    arch_t id = bli_arch_query_id();
    bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

    if (bamdzen == 0)
    {
        if ( cntx == NULL ) cntx = bli_gks_query_cntx();
        const num_t dt = PASTEMAC(s,type);
        float*  x1 ;
        PASTECH(s,dotxf_ker_ft) kfp_df;
        /* Query the context for the kernel function pointer and fusing factor. */
        kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
        b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

            A1 = a + (i  )*rs_at + (0  )*cs_at;
            x1 = x + (0  )*incy;
            y1 = y + (i  )*incy;

            /* y1 = beta * y1 + alpha * A1 * x; */
            kfp_df
            (
            conja,
            conjx,
            n_elem,
            f,
            alpha,
            A1,   cs_at, rs_at,
            x1,   incx,
            beta,
            y1,   incy,
            cntx
            );

        }

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
      return;
    }

    /* Query the context for the kernel function pointer and fusing factor. */
    b_fuse = 8;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (i  )*rs_at + (0  )*cs_at;
        x1 = x + (0  )*incy;
        y1 = y + (i  )*incy;

        /* y1 = beta * y1 + alpha * A1 * x; */
        bli_sdotxf_zen_int_8
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1,   cs_at, rs_at,
          x1,   incx,
          beta,
          y1,   incy,
          cntx
        );

    }
}

INSERT_GENTFUNC_BASIC0_CZ( gemv_unf_var1 )
#else
INSERT_GENTFUNC_BASIC0( gemv_unf_var1 )
#endif
