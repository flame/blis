/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_char* transa, \
       const f77_int*  m, \
       const f77_int*  n, \
       const ftype*    alpha, \
       const ftype*    a, const f77_int* lda, \
       const ftype*    x, const f77_int* incx, \
       const ftype*    beta, \
             ftype*    y, const f77_int* incy  \
     ) \
{ \
    trans_t blis_transa; \
    dim_t   m0, n0; \
    dim_t   m_y, n_x; \
    ftype*  x0; \
    ftype*  y0; \
    inc_t   incx0; \
    inc_t   incy0; \
    inc_t   rs_a, cs_a; \
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
\
    /* Perform BLAS parameter checking. */ \
    PASTEBLACHK(blasname) \
    ( \
      MKSTR(ch), \
      MKSTR(blasname), \
      transa, \
      m, \
      n, \
      lda, \
      incx, \
      incy  \
    ); \
\
    /* Map BLAS chars to their corresponding BLIS enumerated type value. */ \
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa ); \
\
    /* Convert/typecast negative values of m and n to zero. */ \
    bli_convert_blas_dim1( *m, m0 ); \
    bli_convert_blas_dim1( *n, n0 ); \
\
    /* Determine the dimensions of x and y so we can adjust the increments,
       if necessary.*/ \
    bli_set_dims_with_trans( blis_transa, m0, n0, &m_y, &n_x ); \
\
    /* BLAS handles cases where trans(A) has no columns, and x has no elements,
       in a peculiar way. In these situations, BLAS returns without performing
       any action, even though most sane interpretations of gemv would have the
       the operation reduce to y := beta * y. Here, we catch those cases that
       BLAS would normally mishandle and emulate the BLAS exactly so as to
       provide "bug-for-bug" compatibility. Note that this extreme level of
       compatibility would not be as much of an issue if it weren't for the
       fact that some BLAS test suites actually test for these cases. Also, it
       should be emphasized that BLIS, if called natively, does NOT exhibit
       this quirky behavior; it will scale y by beta, as one would expect. */ \
    if ( m_y > 0 && n_x == 0 ) \
    { \
        /* Finalize BLIS. */ \
        bli_finalize_auto(); \
\
        return; \
    } \
\
    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */ \
    bli_convert_blas_incv( n_x, (ftype*)x, *incx, x0, incx0 ); \
    bli_convert_blas_incv( m_y, (ftype*)y, *incy, y0, incy0 ); \
\
    /* Set the row and column strides of A. */ \
    rs_a = 1; \
    cs_a = *lda; \
\
    /* Call BLIS interface. */ \
    PASTEMAC2(ch,blisname,BLIS_TAPI_EX_SUF) \
    ( \
      blis_transa, \
      BLIS_NO_CONJUGATE, \
      m0, \
      n0, \
      (ftype*)alpha, \
      (ftype*)a,  rs_a, cs_a, \
      x0, incx0, \
      (ftype*)beta, \
      y0, incy0, \
      NULL, \
      NULL  \
    ); \
\
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
}


#ifdef BLIS_ENABLE_BLAS
#ifdef BLIS_CONFIG_ZEN2
void dgemv_
     (
       const f77_char* transa,
       const f77_int*  m,
       const f77_int*  n,
       const double*    alpha,
       const double*    a, const f77_int* lda,
       const double*    x, const f77_int* incx,
       const double*    beta,
             double*    y, const f77_int* incy
     )
{
    trans_t blis_transa;
    dim_t   m0, n0;
    dim_t   m_y, n_x;
    double*  x0;
    double*  y0;
    inc_t   incx0;
    inc_t   incy0;
    inc_t   rs_a, cs_a;

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemv)
    (
      MKSTR(d),
      MKSTR(gemv),
      transa,
      m,
      n,
      lda,
      incx,
      incy
    );

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    if      ( *transa == 'n' || *transa == 'N' ) blis_transa = BLIS_NO_TRANSPOSE;
    else if ( *transa == 't' || *transa == 'T' ) blis_transa = BLIS_TRANSPOSE;
    else if ( *transa == 'c' || *transa == 'C' ) blis_transa = BLIS_CONJ_TRANSPOSE;
    else
    {
        // See comment for bli_param_map_netlib_to_blis_side() above.
        //bli_check_error_code( BLIS_INVALID_TRANS );
        blis_transa = BLIS_NO_TRANSPOSE;
    }

    /* Convert/typecast negative values of m and n to zero. */
    if ( *m < 0 ) m0 = ( dim_t )0;
    else              m0 = ( dim_t )(*m);

    if ( *n < 0 ) n0 = ( dim_t )0; 
    else              n0 = ( dim_t )(*n);

    /* Determine the dimensions of x and y so we can adjust the increments,
       if necessary.*/
    if ( bli_does_notrans( blis_transa ) ) { m_y = m0; n_x = n0; }
    else                             { m_y = n0; n_x = m0; }

    /* BLAS handles cases where trans(A) has no columns, and x has no elements,
       in a peculiar way. In these situations, BLAS returns without performing
       any action, even though most sane interpretations of gemv would have the
       the operation reduce to y := beta * y. Here, we catch those cases that
       BLAS would normally mishandle and emulate the BLAS exactly so as to
       provide "bug-for-bug" compatibility. Note that this extreme level of
       compatibility would not be as much of an issue if it weren't for the
       fact that some BLAS test suites actually test for these cases. Also, it
       should be emphasized that BLIS, if called natively, does NOT exhibit
       this quirky behavior; it will scale y by beta, as one would expect. */
    if ( m_y > 0 && n_x == 0 )
    {
        /* Finalize BLIS. */
//      bli_finalize_auto();

        return;
    }

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( *incx < 0 )
    {
        x0    = ((double*)x) + (n_x-1)*(-*incx);
        incx0 = ( inc_t )(*incx);
    }
    else
    {
        x0    = ((double*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((double*)y) + (m_y-1)*(-*incy);
        incy0 = ( inc_t )(*incy);
    }
    else
    {
        y0    = ((double*)y);
        incy0 = ( inc_t )(*incy);
    }

    /* Set the row and column strides of A. */
    rs_a = 1;
    cs_a = *lda;

    /* Call variants based on transpose value. */
    if(bli_does_notrans(blis_transa))
    {
        if((m0 <1200) && (n0 < 1200))
        {
            bli_dgemv_zen_ref_c
            (
                m0,
                n0,
                (double*) alpha,
                (double*)a, cs_a,
                x0, incx0,
                (double*) beta,
                y0, incy0,
                NULL
            );
        }
        else
        {
            //variant_2 is chosen for column-storage
            // and uses axpyf-based implementation
            bli_dgemv_unf_var2
            (
                blis_transa,
                BLIS_NO_CONJUGATE,
                m0,
                n0,
               (double*)alpha,
               (double*)a,  rs_a, cs_a,
               x0, incx0,
               (double*)beta,
               y0, incy0,
               NULL
            );
        }
    }
    else
    {
        //var_1 is chosen for row-storage
        //and uses dotxf-based implementation
        bli_dgemv_unf_var1
        (
        blis_transa,
        BLIS_NO_CONJUGATE,
        m0,
        n0,
        (double*)alpha,
        (double*)a,  rs_a, cs_a,
        x0, incx0,
        (double*)beta,
        y0, incy0,
        NULL
        );

    }

}

void sgemv_
     (
       const f77_char* transa,
       const f77_int*  m,
       const f77_int*  n,
       const float*    alpha,
       const float*    a, const f77_int* lda,
       const float*    x, const f77_int* incx,
       const float*    beta,
             float*    y, const f77_int* incy
     )
{
    trans_t blis_transa;
    dim_t   m0, n0;
    dim_t   m_y, n_x;
    float*  x0;
    float*  y0;
    inc_t   incx0;
    inc_t   incy0;
    inc_t   rs_a, cs_a;

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemv)
    (
      MKSTR(s),
      MKSTR(gemv),
      transa,
      m,
      n,
      lda,
      incx,
      incy
    );

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    if      ( *transa == 'n' || *transa == 'N' ) blis_transa = BLIS_NO_TRANSPOSE;
    else if ( *transa == 't' || *transa == 'T' ) blis_transa = BLIS_TRANSPOSE;
    else if ( *transa == 'c' || *transa == 'C' ) blis_transa = BLIS_CONJ_TRANSPOSE;
    else
    {
        // See comment for bli_param_map_netlib_to_blis_side() above.
        //bli_check_error_code( BLIS_INVALID_TRANS );
        blis_transa = BLIS_NO_TRANSPOSE;
    }

    /* Convert/typecast negative values of m and n to zero. */
    if ( *m < 0 ) m0 = ( dim_t )0;
    else              m0 = ( dim_t )(*m);

    if ( *n < 0 ) n0 = ( dim_t )0; 
    else              n0 = ( dim_t )(*n);

    /* Determine the dimensions of x and y so we can adjust the increments,
       if necessary.*/
    if ( bli_does_notrans( blis_transa ) ) { m_y = m0; n_x = n0; }
    else                             { m_y = n0; n_x = m0; }

    /* BLAS handles cases where trans(A) has no columns, and x has no elements,
       in a peculiar way. In these situations, BLAS returns without performing
       any action, even though most sane interpretations of gemv would have the
       the operation reduce to y := beta * y. Here, we catch those cases that
       BLAS would normally mishandle and emulate the BLAS exactly so as to
       provide "bug-for-bug" compatibility. Note that this extreme level of
       compatibility would not be as much of an issue if it weren't for the
       fact that some BLAS test suites actually test for these cases. Also, it
       should be emphasized that BLIS, if called natively, does NOT exhibit
       this quirky behavior; it will scale y by beta, as one would expect. */
    if ( m_y > 0 && n_x == 0 )
    {
        /* Finalize BLIS. */
//      bli_finalize_auto();

        return;
    }

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( *incx < 0 )
    {
        x0    = ((float*)x) + (n_x-1)*(-*incx);
        incx0 = ( inc_t )(*incx);
    }
    else
    {
        x0    = ((float*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((float*)y) + (m_y-1)*(-*incy);
        incy0 = ( inc_t )(*incy);
    }
    else
    {
        y0    = ((float*)y);
        incy0 = ( inc_t )(*incy);
    }

    /* Set the row and column strides of A. */
    rs_a = 1;
    cs_a = *lda;

    /* Call variants based on transpose value. */
    if(bli_does_notrans(blis_transa))
    {
        bli_sgemv_unf_var2
        (
        blis_transa,
        BLIS_NO_CONJUGATE,
        m0,
        n0,
        (float*)alpha,
        (float*)a,  rs_a, cs_a,
        x0, incx0,
        (float*)beta,
        y0, incy0,
        NULL
        );

    }
    else
    {   
        bli_sgemv_unf_var1
        (
        blis_transa,
        BLIS_NO_CONJUGATE,
        m0,
        n0,
        (float*)alpha,
        (float*)a,  rs_a, cs_a,
        x0, incx0,
        (float*)beta,
        y0, incy0,
        NULL
        );

    }

}

INSERT_GENTFUNC_BLAS_CZ( gemv, gemv )
#else
INSERT_GENTFUNC_BLAS( gemv, gemv )
#endif
#endif
