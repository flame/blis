/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2017, Advanced Micro Devices, Inc.

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

/* ! /brief Double precision dotv function.
*
*  Input:
*       conjx : ignore for real data types
*       n     : input vector length
*       alpha : multiplier
*       x     : pointer to vector
*       incx  : vector increment
*       y     : pointer to vector
*       incy  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       rho   : result
*/
void bli_ddotv_opt_var1
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       double* restrict rho,
       cntx_t*          cntx
     )
{
    double*  restrict x_cast   = x;
    double*  restrict y_cast   = y;
    double*  restrict rho_cast = rho;
    dim_t             i = 0;

    dim_t             n_run;
    dim_t             n_left;

    double*  restrict x1;
    double*  restrict y1;
    double            rho1;
    double            x1c, y1c;

    v4df_t            rho1v;
    v4df_t            x1v, y1v;

    v4df_t            rho2v;
    v4df_t            x2v, y2v;

    v4df_t            rho3v;
    v4df_t            x3v, y3v;

    v4df_t            rho4v;
    v4df_t            x4v, y4v;

    const dim_t       n_elem_per_reg = 4; // AVX 256 bit register holds 4 DP data
    const dim_t       n_iter_unroll  = 4;

    bool_t            use_ref = FALSE;

    // If the vector lengths are zero, set rho to zero and return.
    if ( bli_zero_dim1( n ) )
    {
        PASTEMAC(d,set0s)( *rho_cast );
        return;
    }

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {
        BLIS_DDOTV_KERNEL_REF( conjx,
                               conjy,
                               n,
                               x, incx,
                               y, incy,
                               rho,
                               cntx );
        return;
    }

    n_run       = ( n ) / (n_elem_per_reg * n_iter_unroll);
    n_left      = ( n ) % (n_elem_per_reg * n_iter_unroll);

    x1 = x_cast;
    y1 = y_cast;

    PASTEMAC(d,set0s)( rho1 );

    rho1v.v = _mm256_setzero_pd();
    rho2v.v = _mm256_setzero_pd();
    rho3v.v = _mm256_setzero_pd();
    rho4v.v = _mm256_setzero_pd();

    for ( i = 0; i < n_run; ++i )
    {
        // load the input
        x1v.v = _mm256_loadu_pd( ( double* )x1 );
        y1v.v = _mm256_loadu_pd( ( double* )y1 );

        x2v.v = _mm256_loadu_pd( ( double* )(x1 + n_elem_per_reg));
        y2v.v = _mm256_loadu_pd( ( double* )(y1 + n_elem_per_reg));

        x3v.v = _mm256_loadu_pd( ( double* )(x1 + 2*n_elem_per_reg));
        y3v.v = _mm256_loadu_pd( ( double* )(y1 + 2*n_elem_per_reg));

        x4v.v = _mm256_loadu_pd( ( double* )(x1 + 3*n_elem_per_reg));
        y4v.v = _mm256_loadu_pd( ( double* )(y1 + 3*n_elem_per_reg));
        
        // Calculate the dot product
        rho1v.v += x1v.v * y1v.v;
        rho2v.v += x2v.v * y2v.v;
        rho3v.v += x3v.v * y3v.v;
        rho4v.v += x4v.v * y4v.v;

        x1 += (n_elem_per_reg * n_iter_unroll);
        y1 += (n_elem_per_reg * n_iter_unroll);
    }

    //accumulate the results
    rho1v.v += rho2v.v;
    rho1v.v += rho3v.v;
    rho1v.v += rho4v.v;

    rho1 += rho1v.d[0] + rho1v.d[1] + rho1v.d[2] + rho1v.d[3] ;

    if ( n_left > 0 )
    {
        for ( i = 0; i < n_left; ++i )
        {
            x1c = *x1;
            y1c = *y1;

            rho1 += x1c * y1c;

            x1 += incx;
            y1 += incy;
        }
    }

    PASTEMAC(d,copys)( rho1, *rho_cast ); 
}

/* Union data structure to access AVX registers
*  AVX 256 bit register holds 8 SP data*/
typedef union
{
    __m256 v;
    float  f[8];
} v8ff_t;

/* ! /brief Single precision dotv function.
*
*  Input:
*       conjx : ignore for real data types
*       n     : input vector length
*       alpha : multiplier
*       x     : pointer to vector
*       incx  : vector increment
*       y     : pointer to vector
*       incy  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       rho   : result
*/
void bli_sdotv_opt_var1
     (
       conj_t           conjx,
       conj_t           conjy,
       dim_t            n,
       float* restrict x, inc_t incx,
       float* restrict y, inc_t incy,
       float* restrict rho,
       cntx_t*          cntx
     )
{
    float*  restrict x_cast   = x;
    float*  restrict y_cast   = y;
    float*  restrict rho_cast = rho;
    dim_t             i;

    dim_t             n_run;
    dim_t             n_left;

    float*  restrict x1;
    float*  restrict y1;
    float            rho1;
    float            x1c, y1c;

    v8ff_t            rho1v, rho2v, rho3v, rho4v;
    v8ff_t            x1v, y1v;
    v8ff_t            x2v, y2v;
    v8ff_t            x3v, y3v;
    v8ff_t            x4v, y4v;

    const dim_t       n_elem_per_reg = 8;
    const dim_t       n_iter_unroll  = 4;

    bool_t            use_ref = FALSE;

    // If the vector lengths are zero, set rho to zero and return.
    if ( bli_zero_dim1( n ) )
    {
        PASTEMAC(s,set0s)( *rho_cast );
        return;
    }

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {
        BLIS_SDOTV_KERNEL_REF( conjx,
                               conjy,
                               n,
                               x, incx,
                               y, incy,
                               rho,
                               cntx );
        return;
    }

    n_run       = ( n ) / (n_elem_per_reg * n_iter_unroll);
    n_left      = ( n ) % (n_elem_per_reg * n_iter_unroll);

    x1 = x_cast;
    y1 = y_cast;

    PASTEMAC(s,set0s)( rho1 );


    rho1v.v = _mm256_setzero_ps();
    rho2v.v = _mm256_setzero_ps();
    rho3v.v = _mm256_setzero_ps();
    rho4v.v = _mm256_setzero_ps();

    for ( i = 0; i < n_run; ++i )
    {
        // load the input
        x1v.v = _mm256_loadu_ps( ( float* )x1 );
        y1v.v = _mm256_loadu_ps( ( float* )y1 );

        x2v.v = _mm256_loadu_ps( ( float* )(x1 + n_elem_per_reg));
        y2v.v = _mm256_loadu_ps( ( float* )(y1 + n_elem_per_reg));

        x3v.v = _mm256_loadu_ps( ( float* )(x1 + 2 * n_elem_per_reg));
        y3v.v = _mm256_loadu_ps( ( float* )(y1 + 2 * n_elem_per_reg));

        x4v.v = _mm256_loadu_ps( ( float* )(x1 + 3 * n_elem_per_reg));
        y4v.v = _mm256_loadu_ps( ( float* )(y1 + 3 * n_elem_per_reg));

        // Calculate the dot product
        rho1v.v += x1v.v * y1v.v;
        rho2v.v += x2v.v * y2v.v;
        rho3v.v += x3v.v * y3v.v;
        rho4v.v += x4v.v * y4v.v;

        x1 += (n_elem_per_reg * n_iter_unroll);
        y1 += (n_elem_per_reg * n_iter_unroll);
    }

    //accumulate the results
    rho1v.v += rho2v.v;
    rho1v.v += rho3v.v;
    rho1v.v += rho4v.v;

    rho1 += rho1v.f[0] + rho1v.f[1] + rho1v.f[2] + rho1v.f[3] +
            rho1v.f[4] + rho1v.f[5] + rho1v.f[6] + rho1v.f[7];

    if ( n_left > 0 )
    {
        for ( i = 0; i < n_left; ++i )
        {
            x1c = *x1;
            y1c = *y1;

            rho1 += x1c * y1c;

            x1 += incx;
            y1 += incy;
        }
    }

    PASTEMAC(s,copys)( rho1, *rho_cast );
}
