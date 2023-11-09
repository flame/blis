/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
       uplo_t  uploa, \
       trans_t transa, \
       diag_t  diaga, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx  \
     ) \
{ \
    const num_t dt = PASTEMAC(ch,type); \
\
    bli_init_once(); \
\
    if( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
    ctype*  minus_one  = PASTEMAC(ch,m1); \
    ctype*  A01; \
    ctype*  A11; \
    ctype*  A21; \
    ctype*  a01; \
    ctype*  alpha11; \
    ctype*  a21; \
    ctype*  x0; \
    ctype*  x1; \
    ctype*  x2; \
    ctype*  x01; \
    ctype*  chi11; \
    ctype*  x21; \
    ctype   alpha11_conj; \
    ctype   minus_chi11; \
    dim_t   iter, i, k, j, l; \
    dim_t   b_fuse, f; \
    dim_t   n_ahead, f_ahead; \
    inc_t   rs_at, cs_at; \
    uplo_t  uploa_trans; \
    conj_t  conja; \
\
    /* x = alpha * x; */ \
    PASTEMAC2(ch,scalv,BLIS_TAPI_EX_SUF) \
    ( \
      BLIS_NO_CONJUGATE, \
      m, \
      alpha, \
      x, incx, \
      cntx, \
      NULL  \
    ); \
\
    if      ( bli_does_notrans( transa ) ) \
    { \
        rs_at = rs_a; \
        cs_at = cs_a; \
        uploa_trans = uploa; \
    } \
    else /* if ( bli_does_trans( transa ) ) */ \
    { \
        rs_at = cs_a; \
        cs_at = rs_a; \
        uploa_trans = bli_uplo_toggled( uploa ); \
    } \
\
    conja = bli_extract_conj( transa ); \
\
    PASTECH(ch,axpyf_ker_ft) kfp_af; \
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_AF, cntx ); \
\
    /* We reduce all of the possible cases down to just lower/upper. */ \
    if      ( bli_is_upper( uploa_trans ) ) \
    { \
        for ( iter = 0; iter < m; iter += f ) \
        { \
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse ); \
            i        = m - iter - f; \
            n_ahead  = i; \
            A11      = a + (i  )*rs_at + (i  )*cs_at; \
            A01      = a + (0  )*rs_at + (i  )*cs_at; \
            x1       = x + (i  )*incx; \
            x0       = x + (0  )*incx; \
\
            /* x1 = x1 / triu( A11 ); */ \
            for ( k = 0; k < f; ++k ) \
            { \
                l        = f - k - 1; \
                f_ahead  = l; \
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
                a01      = A11 + (0  )*rs_at + (l  )*cs_at; \
                chi11    = x1  + (l  )*incx; \
                x01      = x1  + (0  )*incx; \
\
                /* chi11 = chi11 / alpha11; */ \
                if ( bli_is_nonunit_diag( diaga ) ) \
                { \
                    PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_conj ); \
                    PASTEMAC(ch,invscals)( alpha11_conj, *chi11 ); \
                } \
\
                /* x01 = x01 - chi11 * a01; */ \
                PASTEMAC(ch,neg2s)( *chi11, minus_chi11 ); \
                if ( bli_is_conj( conja ) ) \
                { \
                    for ( j = 0; j < f_ahead; ++j ) \
                        PASTEMAC(ch,axpyjs)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) ); \
                } \
                else \
                { \
                    for ( j = 0; j < f_ahead; ++j ) \
                        PASTEMAC(ch,axpys)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) ); \
                } \
            } \
\
            /* x0 = x0 - A01 * x1; */ \
            kfp_af \
            ( \
              conja, \
              BLIS_NO_CONJUGATE, \
              n_ahead, \
              f, \
              minus_one, \
              A01, rs_at, cs_at, \
              x1,  incx, \
              x0,  incx, \
              cntx  \
            ); \
        } \
    } \
    else /* if ( bli_is_lower( uploa_trans ) ) */ \
    { \
        for ( iter = 0; iter < m; iter += f ) \
        { \
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse ); \
            i        = iter; \
            n_ahead  = m - iter - f; \
            A11      = a + (i  )*rs_at + (i  )*cs_at; \
            A21      = a + (i+f)*rs_at + (i  )*cs_at; \
            x1       = x + (i  )*incx; \
            x2       = x + (i+f)*incx; \
\
            /* x1 = x1 / tril( A11 ); */ \
            for ( k = 0; k < f; ++k ) \
            { \
                l        = k; \
                f_ahead  = f - k - 1; \
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
                a21      = A11 + (l+1)*rs_at + (l  )*cs_at; \
                chi11    = x1  + (l  )*incx; \
                x21      = x1  + (l+1)*incx; \
\
                /* chi11 = chi11 / alpha11; */ \
                if ( bli_is_nonunit_diag( diaga ) ) \
                { \
                    PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_conj ); \
                    PASTEMAC(ch,invscals)( alpha11_conj, *chi11 ); \
                } \
\
                /* x21 = x21 - chi11 * a21; */ \
                PASTEMAC(ch,neg2s)( *chi11, minus_chi11 ); \
                if ( bli_is_conj( conja ) ) \
                { \
                    for ( j = 0; j < f_ahead; ++j ) \
                        PASTEMAC(ch,axpyjs)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) ); \
                } \
                else \
                { \
                    for ( j = 0; j < f_ahead; ++j ) \
                        PASTEMAC(ch,axpys)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) ); \
                } \
            } \
\
            /* x2 = x2 - A21 * x1; */ \
            kfp_af \
            ( \
              conja, \
              BLIS_NO_CONJUGATE, \
              n_ahead, \
              f, \
              minus_one, \
              A21, rs_at, cs_at, \
              x1,  incx, \
              x2,  incx, \
              cntx  \
            ); \
        } \
    } \
}

void bli_dtrsv_unf_var2
     (
       uplo_t  uploa,
       trans_t transa,
       diag_t  diaga,
       dim_t   m,
       double*  alpha,
       double*  a, inc_t rs_a, inc_t cs_a,
       double*  x, inc_t incx,
       cntx_t* cntx
     )
{

    double*  minus_one  = PASTEMAC(d,m1);
    double*  A01;
    double*  A11;
    double*  A21;
    double*  a01;
    double*  alpha11;
    double*  a21;
    double*  x0;
    double*  x1;
    double*  x2;
    double*  x01;
    double*  chi11;
    double*  x21;
    double   alpha11_conj;
    double   minus_chi11;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_ahead, f_ahead;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here
    bli_init_once();
    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    /* x = alpha * x; */
    PASTEMAC2(d,scalv,BLIS_TAPI_EX_SUF)
    (
      BLIS_NO_CONJUGATE,
      m,
      alpha,
      x, incx,
      cntx,
      NULL
    );

    if ( bli_does_notrans( transa ) )
    {
        rs_at = rs_a;
        cs_at = cs_a;
        uploa_trans = uploa;
    }
    else /* if ( bli_does_trans( transa ) ) */
    {
        rs_at = cs_a;
        cs_at = rs_a;
        uploa_trans = bli_uplo_toggled( uploa );
    }

    conja = bli_extract_conj( transa );

    PASTECH(d,axpyf_ker_ft) kfp_af;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_af = bli_daxpyf_zen_int_16x4;
	    b_fuse = 4;
    }
    else
    {
	    kfp_af = bli_cntx_get_l1f_ker_dt( BLIS_DOUBLE, BLIS_AXPYF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( BLIS_DOUBLE, BLIS_AF, cntx );
    }

    /* We reduce all of the possible cases down to just lower/upper. */
    if      ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_ahead  = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A01      = a + (0  )*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_ahead  = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a01      = A11 + (0  )*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(d,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(d,invscals)( alpha11_conj, *chi11 );
                }

                /* x01 = x01 - chi11 * a01; */
                PASTEMAC(d,neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(d,axpyjs)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(d,axpys)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
            }

            /* x0 = x0 - A01 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A01, rs_at, cs_at,
              x1,  incx,
              x0,  incx,
              cntx
            );
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_ahead  = m - iter - f;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A21      = a + (i+f)*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_ahead  = f - k - 1;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a21      = A11 + (l+1)*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(d,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(d,invscals)( alpha11_conj, *chi11 );
                }

                /* x21 = x21 - chi11 * a21; */
                PASTEMAC(d,neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(d,axpyjs)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(d,axpys)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
            }

            /* x2 = x2 - A21 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A21, rs_at, cs_at,
              x1,  incx,
              x2,  incx,
              cntx
            );
        }
    }
}

void bli_strsv_unf_var2
     (
       uplo_t  uploa,
       trans_t transa,
       diag_t  diaga,
       dim_t   m,
       float*  alpha,
       float*  a, inc_t rs_a, inc_t cs_a,
       float*  x, inc_t incx,
       cntx_t* cntx
     )
{

    float*  minus_one  = PASTEMAC(s, m1);
    float*  A01;
    float*  A11;
    float*  A21;
    float*  a01;
    float*  alpha11;
    float*  a21;
    float*  x0;
    float*  x1;
    float*  x2;
    float*  x01;
    float*  chi11;
    float*  x21;
    float   alpha11_conj;
    float   minus_chi11;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_ahead, f_ahead;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here
    bli_init_once();
    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    /* x = alpha * x; */
    PASTEMAC2(s, scalv,BLIS_TAPI_EX_SUF)
    (
      BLIS_NO_CONJUGATE,
      m,
      alpha,
      x, incx,
      cntx,
      NULL
    );

    if( bli_does_notrans( transa ) )
    {
        rs_at = rs_a;
        cs_at = cs_a;
        uploa_trans = uploa;
    }
    else /* if ( bli_does_trans( transa ) ) */
    {
        rs_at = cs_a;
        cs_at = rs_a;
        uploa_trans = bli_uplo_toggled( uploa );
    }

    conja = bli_extract_conj( transa );

    PASTECH(s, axpyf_ker_ft) kfp_af;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_af = bli_saxpyf_zen_int_5;
	    b_fuse = 5;
    }
    else
    {
	    kfp_af = bli_cntx_get_l1f_ker_dt( BLIS_FLOAT, BLIS_AXPYF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( BLIS_FLOAT, BLIS_AF, cntx );
    }

    /* We reduce all of the possible cases down to just lower/upper. */
    if ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_ahead  = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A01      = a + (0  )*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_ahead  = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a01      = A11 + (0  )*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(s, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(s, invscals)( alpha11_conj, *chi11 );
                }

                /* x01 = x01 - chi11 * a01; */
                PASTEMAC(s, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(s, axpyjs)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(s, axpys)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
            }

            /* x0 = x0 - A01 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A01, rs_at, cs_at,
              x1,  incx,
              x0,  incx,
              cntx
            );
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_ahead  = m - iter - f;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A21      = a + (i+f)*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_ahead  = f - k - 1;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a21      = A11 + (l+1)*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(s, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(s, invscals)( alpha11_conj, *chi11 );
                }

                /* x21 = x21 - chi11 * a21; */
                PASTEMAC(s, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(s, axpyjs)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(s, axpys)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
            }

            /* x2 = x2 - A21 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A21, rs_at, cs_at,
              x1,  incx,
              x2,  incx,
              cntx
            );
        }
    }
}

void bli_ztrsv_unf_var2
     (
       uplo_t  uploa,
       trans_t transa,
       diag_t  diaga,
       dim_t   m,
       dcomplex*  alpha,
       dcomplex*  a, inc_t rs_a, inc_t cs_a,
       dcomplex*  x, inc_t incx,
       cntx_t* cntx
     )
{

    dcomplex*  minus_one  = PASTEMAC(z, m1);
    dcomplex*  A01;
    dcomplex*  A11;
    dcomplex*  A21;
    dcomplex*  a01;
    dcomplex*  alpha11;
    dcomplex*  a21;
    dcomplex*  x0;
    dcomplex*  x1;
    dcomplex*  x2;
    dcomplex*  x01;
    dcomplex*  chi11;
    dcomplex*  x21;
    dcomplex   alpha11_conj;
    dcomplex   minus_chi11;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_ahead, f_ahead;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here
    bli_init_once();
    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    /* x = alpha * x; */
    PASTEMAC2(z, scalv,BLIS_TAPI_EX_SUF)
    (
      BLIS_NO_CONJUGATE,
      m,
      alpha,
      x, incx,
      cntx,
      NULL
    );

    if( bli_does_notrans( transa ) )
    {
        rs_at = rs_a;
        cs_at = cs_a;
        uploa_trans = uploa;
    }
    else /* if ( bli_does_trans( transa ) ) */
    {
        rs_at = cs_a;
        cs_at = rs_a;
        uploa_trans = bli_uplo_toggled( uploa );
    }

    conja = bli_extract_conj( transa );

    PASTECH(z, axpyf_ker_ft) kfp_af;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_af = bli_zaxpyf_zen_int_5;
	    b_fuse = 5;
    }
    else
    {
	    kfp_af = bli_cntx_get_l1f_ker_dt( BLIS_DCOMPLEX, BLIS_AXPYF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_AF, cntx );
    }
    /* We reduce all of the possible cases down to just lower/upper. */
    if      ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_ahead  = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A01      = a + (0  )*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_ahead  = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a01      = A11 + (0  )*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(z, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(z, invscals)( alpha11_conj, *chi11 );
                }

                /* x01 = x01 - chi11 * a01; */
                PASTEMAC(z, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(z, axpyjs)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(z, axpys)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
            }

            /* x0 = x0 - A01 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A01, rs_at, cs_at,
              x1,  incx,
              x0,  incx,
              cntx
            );
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_ahead  = m - iter - f;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A21      = a + (i+f)*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_ahead  = f - k - 1;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a21      = A11 + (l+1)*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(z, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(z, invscals)( alpha11_conj, *chi11 );
                }

                /* x21 = x21 - chi11 * a21; */
                PASTEMAC(z, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(z, axpyjs)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(z, axpys)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
            }

            /* x2 = x2 - A21 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A21, rs_at, cs_at,
              x1,  incx,
              x2,  incx,
              cntx
            );
        }
    }
}

void bli_ctrsv_unf_var2
     (
       uplo_t  uploa,
       trans_t transa,
       diag_t  diaga,
       dim_t   m,
       scomplex*  alpha,
       scomplex*  a, inc_t rs_a, inc_t cs_a,
       scomplex*  x, inc_t incx,
       cntx_t* cntx
     )
{

    scomplex*  minus_one  = PASTEMAC(c, m1);
    scomplex*  A01;
    scomplex*  A11;
    scomplex*  A21;
    scomplex*  a01;
    scomplex*  alpha11;
    scomplex*  a21;
    scomplex*  x0;
    scomplex*  x1;
    scomplex*  x2;
    scomplex*  x01;
    scomplex*  chi11;
    scomplex*  x21;
    scomplex   alpha11_conj;
    scomplex   minus_chi11;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_ahead, f_ahead;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

    // For AMD these APIS are invoked skipping intermediate framework layers
    // Hence we need to ensure that cntx is set here
    bli_init_once();
    if( cntx == NULL ) cntx = bli_gks_query_cntx();

    /* x = alpha * x; */
    PASTEMAC2(c, scalv,BLIS_TAPI_EX_SUF)
    (
      BLIS_NO_CONJUGATE,
      m,
      alpha,
      x, incx,
      cntx,
      NULL
    );

    if( bli_does_notrans( transa ) )
    {
        rs_at = rs_a;
        cs_at = cs_a;
        uploa_trans = uploa;
    }
    else /* if ( bli_does_trans( transa ) ) */
    {
        rs_at = cs_a;
        cs_at = rs_a;
        uploa_trans = bli_uplo_toggled( uploa );
    }

    conja = bli_extract_conj( transa );

    PASTECH(c, axpyf_ker_ft) kfp_af;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_af = bli_caxpyf_zen_int_5;
	    b_fuse = 5;
    }
    else
    {
	    kfp_af = bli_cntx_get_l1f_ker_dt( BLIS_SCOMPLEX, BLIS_AXPYF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_AF, cntx );
    }
    /* We reduce all of the possible cases down to just lower/upper. */
    if      ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_ahead  = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A01      = a + (0  )*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_ahead  = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a01      = A11 + (0  )*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(c, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(c, invscals)( alpha11_conj, *chi11 );
                }

                /* x01 = x01 - chi11 * a01; */
                PASTEMAC(c, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(c, axpyjs)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(c, axpys)( minus_chi11, *(a01 + j*rs_at), *(x01 + j*incx) );
                }
            }

            /* x0 = x0 - A01 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A01, rs_at, cs_at,
              x1,  incx,
              x0,  incx,
              cntx
            );
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_ahead  = m - iter - f;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A21      = a + (i+f)*rs_at + (i  )*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_ahead  = f - k - 1;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a21      = A11 + (l+1)*rs_at + (l  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(c, copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(c, invscals)( alpha11_conj, *chi11 );
                }

                /* x21 = x21 - chi11 * a21; */
                PASTEMAC(c, neg2s)( *chi11, minus_chi11 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(c, axpyjs)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
                else
                {
                    for ( j = 0; j < f_ahead; ++j )
                        PASTEMAC(c, axpys)( minus_chi11, *(a21 + j*rs_at), *(x21 + j*incx) );
                }
            }

            /* x2 = x2 - A21 * x1; */
            kfp_af
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_ahead,
              f,
              minus_one,
              A21, rs_at, cs_at,
              x1,  incx,
              x2,  incx,
              cntx
            );
        }
    }
}
