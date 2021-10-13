/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-21, Advanced Micro Devices, Inc. All rights reserved.

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
#define BLIS_DGEMV_VAR2_FUSE 4

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
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3); \
\
    bli_init_once(); \
\
    if(cntx == NULL) cntx = bli_gks_query_cntx(); \
\
    const num_t dt = PASTEMAC(ch,type); \
\
    ctype*  zero       = PASTEMAC(ch,0); \
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
                                  &n_elem, &n_iter, &rs_at, &cs_at ); \
\
    conja = bli_extract_conj( transa ); \
\
    /* If beta is zero, use setv. Otherwise, scale by beta. */ \
    if ( PASTEMAC(ch,eq0)( *beta ) ) \
    { \
        /* y = 0; */ \
        PASTEMAC2(ch,setv,BLIS_TAPI_EX_SUF) \
        ( \
          BLIS_NO_CONJUGATE, \
          n_elem, \
          zero, \
          y, incy, \
          cntx, \
          NULL  \
        ); \
    } \
    else \
    { \
        /* y = beta * y; */ \
        PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
        ( \
          BLIS_NO_CONJUGATE, \
          n_elem, \
          beta, \
          y, incy, \
          cntx, \
          NULL  \
        ); \
    } \
\
    PASTECH(ch,axpyf_ker_ft) kfp_af; \
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx ); \
\
    for ( i = 0; i < n_iter; i += f ) \
    { \
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse ); \
\
        A1 = a + (0  )*rs_at + (i  )*cs_at; \
        x1 = x + (i  )*incx; \
        y1 = y + (0  )*incy; \
\
        /* y = y + alpha * A1 * x1; */ \
        kfp_af \
        ( \
          conja, \
          conjx, \
          n_elem, \
          f, \
          alpha, \
          A1, rs_at, cs_at, \
          x1, incx, \
          y1, incy, \
          cntx  \
        ); \
    } \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3); \
}

#ifdef BLIS_CONFIG_EPYC

void bli_dgemv_unf_var2
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

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    double*  A1;
    double*  x1;
    dim_t   i;
    dim_t   f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;
    //memory pool declarations for packing vector Y.
    mem_t   mem_bufY;
    rntm_t  rntm;
    double  *y_buf = y;
    inc_t   buf_incy = incy;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is hadled by scalv internally */

    bli_dscalv_zen_int10
    (
        BLIS_NO_CONJUGATE,
        n_elem,
        beta,
        y, incy,
        NULL
    );

    if( bli_deq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    if (incy > 1)
    {
        /*
          Initialize mem pool buffer to NULL and size to 0
          "buf" and "size" fields are assigned once memory
          is allocated from the pool in bli_membrk_acquire_m().
          This will ensure bli_mem_is_alloc() will be passed on
          an allocated memory if created or a NULL .
        */
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;

        /* In order to get the buffer from pool via rntm access to memory broker
        is needed.Following are initializations for rntm */

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_membrk_rntm_set_membrk( &rntm );

        //calculate the size required for n_elem double elements in vector Y.
        size_t buffer_size = n_elem * sizeof(double);

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_unf_var2(): get mem pool block\n" );
        #endif

        /*acquire a Buffer(n_elem*size(double)) from the memory broker
        and save the associated mem_t entry to mem_bufY.*/
        bli_membrk_acquire_m(&rntm,
                                buffer_size,
                                BLIS_BUFFER_FOR_B_PANEL,
                                &mem_bufY);

        /*Continue packing Y if buffer memory is allocated*/
        if ((bli_mem_is_alloc( &mem_bufY )))
        {
            y_buf = bli_mem_buffer(&mem_bufY);

            //pack Y vector with non-unit stride to a temp buffer y_buf with unit stride
            for(dim_t y_index = 0 ; y_index < n_elem ; y_index++)
            {
                *(y_buf + y_index) =  *(y + (y_index * incy)) ;
            }
            // stride of vector y_buf =1
            buf_incy = 1;
        }
    }

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, BLIS_DGEMV_VAR2_FUSE );

        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;

        /* y = y + alpha * A1 * x1; */
        bli_daxpyf_zen_int_16x4
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1, rs_at, cs_at,
          x1, incx,
          y_buf, buf_incy,
          NULL
        );
    }
    if ((incy > 1) && bli_mem_is_alloc( &mem_bufY ))
    {
        //store the result from unit strided y_buf to non-unit strided Y
        for(dim_t y_index = 0 ; y_index < n_elem ; y_index++)
        {
            *(y + (y_index * incy)) = *(y_buf + y_index) ;
        }

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dgemv_unf_var2(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool
        bli_membrk_release(&rntm , &mem_bufY);
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

void bli_sgemv_unf_var2
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

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    float*  A1;
    float*  x1;
    float*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is hadled by scalv internally */

    bli_sscalv_zen_int10
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y, incy,
      NULL
    );

    if( bli_seq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    /* Query the context for the kernel function pointer and fusing factor. */
    b_fuse = 6;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;
        y1 = y + (0  )*incy;

        /* y = y + alpha * A1 * x1; */
        bli_saxpyf_zen_int_6
        (
          conja,
          conjx,
          n_elem,
          f,
          alpha,
          A1, rs_at, cs_at,
          x1, incx,
          y1, incy,
          NULL
        );
    }
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}


void bli_zgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       dcomplex*  alpha,
       dcomplex*  a, inc_t rs_a, inc_t cs_a,
       dcomplex*  x, inc_t incx,
       dcomplex*  beta,
       dcomplex*  y, inc_t incy,
       cntx_t* cntx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    dcomplex*  A1;
    dcomplex*  x1;
    dcomplex*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is hadled by scalv internally */
/*    bli_zscalv_zen_int10
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y,
      incy,
      cntx
    );*/
    bli_zscalv_ex
	    (
	     BLIS_NO_CONJUGATE,
	     n_elem,
	     beta,
	     y, incy,
	     cntx,
	     NULL
	    );

    if( bli_zeq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
        return;
    }

    // for non-unit incx, incy and rs_at and conjugate will be added in the next patch
    if( (incx == 1 && incy == 1 && rs_at == 1 ) &&
         !bli_is_conj(conja) && !bli_is_conj(conjx) && !bli_is_trans(transa))
    {
        // This gemv code deals with the followint conditions only
        // 1. incx, incy, and row stride equal to one
        // 2. Non conjugate A matrix and X vector
        // 3. No Transpose for A Martix
        // Rest is taken care by the else part (axpyf implementation)
        bli_zgemv_zen_int_4x4
        (
            conja,
            conjx,
            m,
            n,
            alpha,
            a, rs_at, cs_at,
            x, incx,
            beta,
            y, incy,
            NULL
        );
    }
    else
    {
        /* fusing factor */
        b_fuse = 4;

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );
            A1 = a + (0  )*rs_at + (i  )*cs_at;
            x1 = x + (i  )*incx;
            y1 = y + (0  )*incy;

            /* y = y + alpha * A1 * x1; */
            bli_zaxpyf_zen_int_4
            (
                conja,
                conjx,
                n_elem,
                f,
                alpha,
                A1, rs_at, cs_at,
                x1, incx,
                y1, incy,
                NULL
            );
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}

void bli_cgemv_unf_var2
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       scomplex*  alpha,
       scomplex*  a, inc_t rs_a, inc_t cs_a,
       scomplex*  x, inc_t incx,
       scomplex*  beta,
       scomplex*  y, inc_t incy,
       cntx_t* cntx
     )
{

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);
    scomplex*  A1;
    scomplex*  x1;
    scomplex*  y1;
    dim_t   i;
    dim_t   b_fuse, f;
    dim_t   n_elem, n_iter;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &n_elem, &n_iter, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    /* If beta is zero, use setv. Otherwise, scale by beta. */
        /* y = beta * y; */
    /* beta=0 case is hadled by scalv internally */
    /*bli_cscalv_zen_int10
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y,
      incy,
      cntx
    );*/
    bli_cscalv_ex
	    (
	     BLIS_NO_CONJUGATE,
	     n_elem,
	     beta,
	     y, incy,
	     cntx,
	     NULL
	    );



    if( bli_ceq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    // for non-unit incx, incy and rs_at and conjugate will be added in the next patch
    if( ( (incx == 1) && (incy == 1) && (rs_at == 1) ) &&
         !bli_is_conj(conja) && !bli_is_conj(conjx) &&
         !bli_is_trans(transa))
    {
        // This gemv code deals with the followint conditions only
        // 1. incx, incy, and row stride equal to one
        // 2. Non conjugate A matrix and X vector
        // 3. No Transpose for A Martix
        // Rest is taken care by the else part (axpyf implementation)
        bli_cgemv_zen_int_4x4
        (
            conja,
            conjx,
            m,
            n,
            alpha,
            a, rs_at, cs_at,
            x, incx,
            beta,
            y, incy,
            NULL
        );
    }
    else
    {
        /* fusing factor. */
        b_fuse = 4;

        for ( i = 0; i < n_iter; i += f )
        {
            f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );
            A1 = a + (0  )*rs_at + (i  )*cs_at;
            x1 = x + (i  )*incx;
            y1 = y + (0  )*incy;

            /* y = y + alpha * A1 * x1; */
            bli_caxpyf_zen_int_4
            (
              conja,
              conjx,
              n_elem,
              f,
              alpha,
              A1, rs_at, cs_at,
              x1, incx,
              y1, incy,
              NULL
            );
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}


#else
INSERT_GENTFUNC_BASIC0( gemv_unf_var2 )
#endif
