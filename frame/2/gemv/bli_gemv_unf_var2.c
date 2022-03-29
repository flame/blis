/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

    double*  A1;
    double*  x1;
    double*  y1;
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

        bli_dscalv_zen_int10
        (
          BLIS_NO_CONJUGATE,
          n_elem,
          beta,
          y, incy,
          NULL
        );

    /* Query the context for the kernel function pointer and fusing factor. */
    b_fuse = 5;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;
        y1 = y + (0  )*incy;

        /* y = y + alpha * A1 * x1; */
        bli_daxpyf_zen_int_5
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

    /* Query the context for the kernel function pointer and fusing factor. */
    b_fuse = 5;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );

        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;
        y1 = y + (0  )*incy;

        /* y = y + alpha * A1 * x1; */
        bli_saxpyf_zen_int_5
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
	return;
    }

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
    bli_cscalv_ex
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      beta,
      y, incy,
      cntx,
      NULL
    );

    /* fusing factor. */
    b_fuse = 5;

    for ( i = 0; i < n_iter; i += f )
    {
        f  = bli_determine_blocksize_dim_f( i, n_iter, b_fuse );
        A1 = a + (0  )*rs_at + (i  )*cs_at;
        x1 = x + (i  )*incx;
        y1 = y + (0  )*incy;

        /* y = y + alpha * A1 * x1; */
        bli_caxpyf_zen_int_5
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


#else
INSERT_GENTFUNC_BASIC0( gemv_unf_var2 )
#endif
