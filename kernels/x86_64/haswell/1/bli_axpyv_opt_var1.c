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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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
#include "immintrin.h"

/* Union data structure to access AVX registers
*  AVX 256 bit register holds 4 DP data*/
typedef union
{
    __m256d v;
    double  d[4];
} v4df_t;

/* ! /brief Double precision axpyv function.
*
*  Input:
*       conjx : ignore for real data types
*       n     : input vector length
*       alpha : multiplier
*       x     : pointer to vector
*       incx  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       y     : pointer to vector
*       incy  : vector increment
*/
void bli_daxpyv_opt_var1
     (
       conj_t           conjx,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t*          cntx
     )
{
    double*  restrict alpha_cast = alpha;
    double*  restrict x_cast = x;
    double*  restrict y_cast = y;
    dim_t             i;

    const dim_t       n_elem_per_reg = 4;
    const dim_t       n_iter_unroll  = 4;

    dim_t             n_run;
    dim_t             n_left;

    double*  restrict x1;
    double*  restrict y1;
    double            alpha1c, x1c;

    v4df_t            alpha1v;
    v4df_t            x1v, x2v, x3v, x4v;
    v4df_t            y1v, y2v, y3v, y4v;

    bool_t            use_ref = FALSE;


    if ( bli_zero_dim1( n ) ) return;

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

    //use_ref = TRUE;
    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {
        BLIS_DAXPYV_KERNEL_REF( conjx,
                                n,
                                alpha,
                                x, incx,
                                y, incy,
                                cntx );
        return;
    }


    n_run       = ( n  ) / ( n_elem_per_reg * n_iter_unroll );
    n_left      = ( n  ) % ( n_elem_per_reg * n_iter_unroll );

    alpha1c = *alpha_cast;

    x1 = x_cast;
    y1 = y_cast;

    // broadcast scalar multiple to vector registers
    alpha1v.v = _mm256_broadcast_sd(&(alpha1c));

    for ( i = 0; i < n_run; ++i )
    {
         // load the input
         y1v.v = _mm256_loadu_pd( ( double* )y1 );
         x1v.v = _mm256_loadu_pd( ( double* )x1 );

         y2v.v = _mm256_loadu_pd( ( double* )(y1 + 4) );
         x2v.v = _mm256_loadu_pd( ( double* )(x1 + 4) );

         y3v.v = _mm256_loadu_pd( ( double* )(y1 + 8) );
         x3v.v = _mm256_loadu_pd( ( double* )(x1 + 8) );

         y4v.v = _mm256_loadu_pd( ( double* )(y1 + 12) );
         x4v.v = _mm256_loadu_pd( ( double* )(x1 + 12) );
                
         // perform : y += alpha * x;
         y1v.v += alpha1v.v * x1v.v;
         y2v.v += alpha1v.v * x2v.v;
         y3v.v += alpha1v.v * x3v.v;            
         y4v.v += alpha1v.v * x4v.v;

        // store the output.        
        _mm256_storeu_pd( ( double* )(y1    ), y1v.v );
        _mm256_storeu_pd( ( double* )(y1 + 4), y2v.v );
        _mm256_storeu_pd( ( double* )(y1 + 8), y3v.v );
        _mm256_storeu_pd( ( double* )(y1 + 12),y4v.v );

        x1 += n_elem_per_reg * n_iter_unroll;
        y1 += n_elem_per_reg * n_iter_unroll;
    }
    
    // if input data size is non multiple of the number of elements in vector register
    if ( n_left > 0 )
    {
        for ( i = 0; i < n_left; ++i )
        {
            x1c = *x1;

            *y1 += alpha1c * x1c;

            x1 += incx;
            y1 += incy;
        }
    }
}

/* Union data structure to access AVX registers
*  AVX 256 bit register holds 8 SP data*/
typedef union
{
    __m256 v;
    float  f[8];
} v8ff_t;

/* ! /brief Single precision axpyv function.
*
*  Input:
*       conjx : ignore for real data types
*       n     : input vector length
*       alpha : multiplier
*       x     : pointer to vector
*       incx  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       y     : pointer to vector
*       incy  : vector increment
*/
void bli_saxpyv_opt_var1
     (
       conj_t           conjx,
       dim_t            n,
       float* restrict alpha,
       float* restrict x, inc_t incx,
       float* restrict y, inc_t incy,
       cntx_t*          cntx
     )
{
    float*  restrict alpha_cast = alpha;
    float*  restrict x_cast = x;
    float*  restrict y_cast = y;
    dim_t             i;

    const dim_t       n_elem_per_reg = 8;
    const dim_t       n_iter_unroll  = 4;

    dim_t             n_run;
    dim_t             n_left;

    float*  restrict x1;
    float*  restrict y1;
    float            alpha1c, x1c;

    v8ff_t            alpha1v;
    v8ff_t            x1v, x2v, x3v, x4v;
    v8ff_t            y1v, y2v, y3v, y4v;

    bool_t            use_ref = FALSE;


    if ( bli_zero_dim1( n ) ) return;

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {
        BLIS_SAXPYV_KERNEL_REF( conjx,
                                n,
                                alpha,
                                x, incx,
                                y, incy,
                                cntx );
        return;
    }


    n_run       = ( n  ) / ( n_elem_per_reg * n_iter_unroll );
    n_left      = ( n  ) % ( n_elem_per_reg * n_iter_unroll );

    alpha1c = *alpha_cast;

    x1 = x_cast;
    y1 = y_cast;

    alpha1v.v = _mm256_broadcast_ss(&alpha1c );

    for ( i = 0; i < n_run; ++i )
    {
        y1v.v = _mm256_loadu_ps( ( float* )y1 );
        x1v.v = _mm256_loadu_ps( ( float* )x1 );

        y2v.v = _mm256_loadu_ps( ( float* )(y1 + 8) );
        x2v.v = _mm256_loadu_ps( ( float* )(x1 + 8) );

        y3v.v = _mm256_loadu_ps( ( float* )(y1 + 16) );
        x3v.v = _mm256_loadu_ps( ( float* )(x1 + 16) );

        y4v.v = _mm256_loadu_ps( ( float* )(y1 + 24) );
        x4v.v = _mm256_loadu_ps( ( float* )(x1 + 24) );

        y1v.v += alpha1v.v * x1v.v;
        y2v.v += alpha1v.v * x2v.v;
        y3v.v += alpha1v.v * x3v.v;
        y4v.v += alpha1v.v * x4v.v;

        _mm256_storeu_ps( ( float* )(y1    ), y1v.v );
        _mm256_storeu_ps( ( float* )(y1 + 8), y2v.v );
        _mm256_storeu_ps( ( float* )(y1 + 16), y3v.v );
        _mm256_storeu_ps( ( float* )(y1 + 24), y4v.v );

        x1 += n_elem_per_reg * n_iter_unroll;
        y1 += n_elem_per_reg * n_iter_unroll;
    }
    
    // if input data size is non multiple of the number of elements in vector register
    if ( n_left > 0 )
    {
        for ( i = 0; i < n_left; ++i )
        {
            x1c = *x1;

            *y1 += alpha1c * x1c;

            x1 += incx;
            y1 += incy;
        }
    }
}
