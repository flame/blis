/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019-2023, Advanced Micro Devices, Inc. All rights reserved.

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
    if(cntx == NULL) cntx = bli_gks_query_cntx(); \
    const num_t dt = PASTEMAC(ch,type); \
\
    ctype*  one        = PASTEMAC(ch,1); \
    ctype*  minus_one  = PASTEMAC(ch,m1); \
    ctype*  A10; \
    ctype*  A11; \
    ctype*  A12; \
    ctype*  a10t; \
    ctype*  alpha11; \
    ctype*  a12t; \
    ctype*  x0; \
    ctype*  x1; \
    ctype*  x2; \
    ctype*  x01; \
    ctype*  chi11; \
    ctype*  x21; \
    ctype   alpha11_conj; \
    ctype   rho1; \
    dim_t   iter, i, k, j, l; \
    dim_t   b_fuse, f; \
    dim_t   n_behind, f_behind; \
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
    PASTECH(ch,dotxf_ker_ft) kfp_df; \
\
    /* Query the context for the kernel function pointer and fusing factor. */ \
    kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
    /* We reduce all of the possible cases down to just lower/upper. */ \
    if      ( bli_is_upper( uploa_trans ) ) \
    { \
        for ( iter = 0; iter < m; iter += f ) \
        { \
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse ); \
            i        = m - iter - f; \
            n_behind = iter; \
            A11      = a + (i  )*rs_at + (i  )*cs_at; \
            A12      = a + (i  )*rs_at + (i+f)*cs_at; \
            x1       = x + (i  )*incx; \
            x2       = x + (i+f)*incx; \
\
            /* x1 = x1 - A12 * x2; */ \
            kfp_df \
            ( \
              conja, \
              BLIS_NO_CONJUGATE, \
              n_behind, \
              f, \
              minus_one, \
              A12, cs_at, rs_at, \
              x2,  incx, \
              one, \
              x1,  incx, \
              cntx  \
            ); \
\
            /* x1 = x1 / triu( A11 ); */ \
            for ( k = 0; k < f; ++k ) \
            { \
                l        = f - k - 1; \
                f_behind = k; \
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
                a12t     = A11 + (l  )*rs_at + (l+1)*cs_at; \
                chi11    = x1  + (l  )*incx; \
                x21      = x1  + (l+1)*incx; \
\
                /* chi11 = chi11 - a12t * x21; */ \
                PASTEMAC(ch,set0s)( rho1 ); \
                if ( bli_is_conj( conja ) ) \
                { \
                    for ( j = 0; j < f_behind; ++j ) \
                        PASTEMAC(ch,dotjs)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 ); \
                } \
                else \
                { \
                    for ( j = 0; j < f_behind; ++j ) \
                        PASTEMAC(ch,dots)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 ); \
                } \
                PASTEMAC(ch,subs)( rho1, *chi11 ); \
\
                /* chi11 = chi11 / alpha11; */ \
                if ( bli_is_nonunit_diag( diaga ) ) \
                { \
                    PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_conj ); \
                    PASTEMAC(ch,invscals)( alpha11_conj, *chi11 ); \
                } \
            } \
        } \
    } \
    else /* if ( bli_is_lower( uploa_trans ) ) */ \
    { \
        for ( iter = 0; iter < m; iter += f ) \
        { \
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse ); \
            i        = iter; \
            n_behind = i; \
            A11      = a + (i  )*rs_at + (i  )*cs_at; \
            A10      = a + (i  )*rs_at + (0  )*cs_at; \
            x1       = x + (i  )*incx; \
            x0       = x + (0  )*incx; \
\
            /* x1 = x1 - A10 * x0; */ \
            kfp_df \
            ( \
              conja, \
              BLIS_NO_CONJUGATE, \
              n_behind, \
              f, \
              minus_one, \
              A10, cs_at, rs_at, \
              x0,  incx, \
              one, \
              x1,  incx, \
              cntx  \
            ); \
\
            /* x1 = x1 / tril( A11 ); */ \
            for ( k = 0; k < f; ++k ) \
            { \
                l        = k; \
                f_behind = l; \
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at; \
                a10t     = A11 + (l  )*rs_at + (0  )*cs_at; \
                chi11    = x1  + (l  )*incx; \
                x01      = x1  + (0  )*incx; \
\
                /* chi11 = chi11 - a10t * x01; */ \
                PASTEMAC(ch,set0s)( rho1 ); \
                if ( bli_is_conj( conja ) ) \
                { \
                    for ( j = 0; j < f_behind; ++j ) \
                        PASTEMAC(ch,dotjs)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 ); \
                } \
                else \
                { \
                    for ( j = 0; j < f_behind; ++j ) \
                        PASTEMAC(ch,dots)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 ); \
                } \
                PASTEMAC(ch,subs)( rho1, *chi11 ); \
\
                /* chi11 = chi11 / alpha11; */ \
                if ( bli_is_nonunit_diag( diaga ) ) \
                { \
                    PASTEMAC(ch,copycjs)( conja, *alpha11, alpha11_conj ); \
                    PASTEMAC(ch,invscals)( alpha11_conj, *chi11 ); \
                } \
            } \
        } \
    } \
}

void bli_dtrsv_unf_var1
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

    double*  one        = PASTEMAC(d,1);
    double*  minus_one  = PASTEMAC(d,m1);
    double*  A10;
    double*  A11;
    double*  A12;
    double*  a10t;
    double*  alpha11;
    double*  a12t;
    double*  x0;
    double*  x1;
    double*  x2;
    double*  x01;
    double*  chi11;
    double*  x21;
    double   alpha11_conj;
    double   rho1;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_behind, f_behind;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

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

    PASTECH(d,dotxf_ker_ft) kfp_df;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_df = bli_ddotxf_zen_int_8;
	    b_fuse = 8;
    }
    else
    {
	    if ( cntx == NULL ) cntx = bli_gks_query_cntx();
	    num_t dt = PASTEMAC(d,type);
	    kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );
    }

    /* We reduce all of the possible cases down to just lower/upper. */
    if      ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_behind = iter;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A12      = a + (i  )*rs_at + (i+f)*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 - A12 * x2; */
            kfp_df
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_behind,
              f,
              minus_one,
              A12, cs_at, rs_at,
              x2,  incx,
              one,
              x1,  incx,
              cntx
            );

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_behind = k;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a12t     = A11 + (l  )*rs_at + (l+1)*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 - a12t * x21; */
                PASTEMAC(d,set0s)( rho1 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(d,dotjs)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 );
                }
                else
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(d,dots)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 );
                }
                PASTEMAC(d,subs)( rho1, *chi11 );

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(d,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(d,invscals)( alpha11_conj, *chi11 );
                }
            }
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_behind = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A10      = a + (i  )*rs_at + (0  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 - A10 * x0; */
            kfp_df
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_behind,
              f,
              minus_one,
              A10, cs_at, rs_at,
              x0,  incx,
              one,
              x1,  incx,
              cntx
            );

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_behind = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a10t     = A11 + (l  )*rs_at + (0  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 - a10t * x01; */
                PASTEMAC(d,set0s)( rho1 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(d,dotjs)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 );
                }
                else
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(d,dots)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 );
                }
                PASTEMAC(d,subs)( rho1, *chi11 );

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(d,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(d,invscals)( alpha11_conj, *chi11 );
                }
            }
        }
    }
}

void bli_strsv_unf_var1
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

    float*  one        = PASTEMAC(s,1);
    float*  minus_one  = PASTEMAC(s,m1);
    float*  A10;
    float*  A11;
    float*  A12;
    float*  a10t;
    float*  alpha11;
    float*  a12t;
    float*  x0;
    float*  x1;
    float*  x2;
    float*  x01;
    float*  chi11;
    float*  x21;
    float   alpha11_conj;
    float   rho1;
    dim_t   iter, i, k, j, l;
    dim_t   b_fuse, f;
    dim_t   n_behind, f_behind;
    inc_t   rs_at, cs_at;
    uplo_t  uploa_trans;
    conj_t  conja;

    /* x = alpha * x; */
    PASTEMAC2(s,scalv,BLIS_TAPI_EX_SUF)
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

    PASTECH(s,dotxf_ker_ft) kfp_df;

    // This function is invoked on all architectures including 'generic'.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    kfp_df = bli_sdotxf_zen_int_8;
	    b_fuse = 8;
    }
    else
    {
	    if ( cntx == NULL ) cntx = bli_gks_query_cntx();
	    num_t dt = PASTEMAC(s,type);
	    kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
	    b_fuse = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx );

    }

    /* We reduce all of the possible cases down to just lower/upper. */
    if      ( bli_is_upper( uploa_trans ) )
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_b( iter, m, b_fuse );
            i        = m - iter - f;
            n_behind = iter;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A12      = a + (i  )*rs_at + (i+f)*cs_at;
            x1       = x + (i  )*incx;
            x2       = x + (i+f)*incx;

            /* x1 = x1 - A12 * x2; */
            kfp_df
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_behind,
              f,
              minus_one,
              A12, cs_at, rs_at,
              x2,  incx,
              one,
              x1,  incx,
              cntx
            );

            /* x1 = x1 / triu( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = f - k - 1;
                f_behind = k;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a12t     = A11 + (l  )*rs_at + (l+1)*cs_at;
                chi11    = x1  + (l  )*incx;
                x21      = x1  + (l+1)*incx;

                /* chi11 = chi11 - a12t * x21; */
                PASTEMAC(s,set0s)( rho1 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(s,dotjs)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 );
                }
                else
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(s,dots)( *(a12t + j*cs_at), *(x21 + j*incx), rho1 );
                }
                PASTEMAC(s,subs)( rho1, *chi11 );

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(s,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(s,invscals)( alpha11_conj, *chi11 );
                }
            }
        }
    }
    else /* if ( bli_is_lower( uploa_trans ) ) */
    {
        for ( iter = 0; iter < m; iter += f )
        {
            f        = bli_determine_blocksize_dim_f( iter, m, b_fuse );
            i        = iter;
            n_behind = i;
            A11      = a + (i  )*rs_at + (i  )*cs_at;
            A10      = a + (i  )*rs_at + (0  )*cs_at;
            x1       = x + (i  )*incx;
            x0       = x + (0  )*incx;

            /* x1 = x1 - A10 * x0; */
            kfp_df
            (
              conja,
              BLIS_NO_CONJUGATE,
              n_behind,
              f,
              minus_one,
              A10, cs_at, rs_at,
              x0,  incx,
              one,
              x1,  incx,
              cntx
            );

            /* x1 = x1 / tril( A11 ); */
            for ( k = 0; k < f; ++k )
            {
                l        = k;
                f_behind = l;
                alpha11  = A11 + (l  )*rs_at + (l  )*cs_at;
                a10t     = A11 + (l  )*rs_at + (0  )*cs_at;
                chi11    = x1  + (l  )*incx;
                x01      = x1  + (0  )*incx;

                /* chi11 = chi11 - a10t * x01; */
                PASTEMAC(s,set0s)( rho1 );
                if ( bli_is_conj( conja ) )
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(s,dotjs)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 );
                }
                else
                {
                    for ( j = 0; j < f_behind; ++j )
                        PASTEMAC(s,dots)( *(a10t + j*cs_at), *(x01 + j*incx), rho1 );
                }
                PASTEMAC(s,subs)( rho1, *chi11 );

                /* chi11 = chi11 / alpha11; */
                if ( bli_is_nonunit_diag( diaga ) )
                {
                    PASTEMAC(s,copycjs)( conja, *alpha11, alpha11_conj );
                    PASTEMAC(s,invscals)( alpha11_conj, *chi11 );
                }
            }
        }
    }
}

INSERT_GENTFUNC_BASIC0_CZ( trsv_unf_var1 )

