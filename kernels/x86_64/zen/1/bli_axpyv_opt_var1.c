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
  float  f[8] __attribute__((aligned(64)));
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
}// end of function


void bli_saxpyv_opt_var10  (
			     conj_t           conjx,
			     dim_t            n,
			     float* restrict alpha,
			     float* restrict x, inc_t incx,
			     float* restrict y, inc_t incy,
                             cntx_t*          cntx
			    )
{
  float* restrict x1 = x;
  float* restrict y1 = y;
  dim_t  i;
  //  dim_t j;
  __m256  alpha1v;
  __m256 xv[10];
  __m256 yv[10];
  __m256 zv[10];
  
  if ( ( (n) == 0 ) ) return;

  /* If alpha is zero, return. */
  if ( ( ((*alpha)) == (0.0F) ) ) return;


  if ( incx == 1 && incy == 1 )
    {
      alpha1v = _mm256_broadcast_ss( alpha );
     
      for (i = 0; (i + 79) < n; i += 80)
        {
          // 80 elements will be processed per loop. 10 FMAs will run per loop
          xv[0] = _mm256_loadu_ps( ( float* )(x1 + 0*8 ) );
          xv[1] = _mm256_loadu_ps( ( float* )(x1 + 1*8 ) );
          xv[2] = _mm256_loadu_ps( ( float* )(x1 + 2*8 ) );
          xv[3] = _mm256_loadu_ps( ( float* )(x1 + 3*8 ) );
	  xv[4] = _mm256_loadu_ps( ( float* )(x1 + 4*8 ) );
          xv[5] = _mm256_loadu_ps( ( float* )(x1 + 5*8 ) );
          xv[6] = _mm256_loadu_ps( ( float* )(x1 + 6*8 ) );
          xv[7] = _mm256_loadu_ps( ( float* )(x1 + 7*8 ) );
          xv[8] = _mm256_loadu_ps( ( float* )(x1 + 8*8 ) );
          xv[9] = _mm256_loadu_ps( ( float* )(x1 + 9*8 ) );

          yv[0] = _mm256_loadu_ps( ( float* )(y1 + 0*8 ) );
          yv[1] = _mm256_loadu_ps( ( float* )(y1 + 1*8 ) );
          yv[2] = _mm256_loadu_ps( ( float* )(y1 + 2*8 ) );
          yv[3] = _mm256_loadu_ps( ( float* )(y1 + 3*8 ) );
          yv[4] = _mm256_loadu_ps( ( float* )(y1 + 4*8 ) );
          yv[5] = _mm256_loadu_ps( ( float* )(y1 + 5*8 ) );
          yv[6] = _mm256_loadu_ps( ( float* )(y1 + 6*8 ) );
          yv[7] = _mm256_loadu_ps( ( float* )(y1 + 7*8 ) );
          yv[8] = _mm256_loadu_ps( ( float* )(y1 + 8*8 ) );
          yv[9] = _mm256_loadu_ps( ( float* )(y1 + 9*8 ) );

          zv[0] =  _mm256_fmadd_ps(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_ps(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_ps(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_ps(xv[3],  alpha1v, yv[3]);
          zv[4] =  _mm256_fmadd_ps(xv[4],  alpha1v, yv[4]);
          zv[5] =  _mm256_fmadd_ps(xv[5],  alpha1v, yv[5]);
          zv[6] =  _mm256_fmadd_ps(xv[6],  alpha1v, yv[6]);
          zv[7] =  _mm256_fmadd_ps(xv[7],  alpha1v, yv[7]);
          zv[8] =  _mm256_fmadd_ps(xv[8],  alpha1v, yv[8]);
          zv[9] =  _mm256_fmadd_ps(xv[9],  alpha1v, yv[9]);

          _mm256_storeu_ps( ( float* )(y1 + 0*8 ), zv[0] );
          _mm256_storeu_ps( ( float* )(y1 + 1*8 ), zv[1] );
          _mm256_storeu_ps( ( float* )(y1 + 2*8 ), zv[2] );
          _mm256_storeu_ps( ( float* )(y1 + 3*8 ), zv[3] );
          _mm256_storeu_ps( ( float* )(y1 + 4*8 ), zv[4] );
          _mm256_storeu_ps( ( float* )(y1 + 5*8 ), zv[5] );
          _mm256_storeu_ps( ( float* )(y1 + 6*8 ), zv[6] );
          _mm256_storeu_ps( ( float* )(y1 + 7*8 ), zv[7] );
          _mm256_storeu_ps( ( float* )(y1 + 8*8 ), zv[8] );
          _mm256_storeu_ps( ( float* )(y1 + 9*8 ), zv[9] );

          x1 += 80;
          y1 += 80;
        }
     
      for ( ; (i + 39) < n; i += 40 )
        {
          xv[0] = _mm256_loadu_ps( ( float* )(x1 + 0*8 ) );
          xv[1] = _mm256_loadu_ps( ( float* )(x1 + 1*8 ) );
          xv[2] = _mm256_loadu_ps( ( float* )(x1 + 2*8 ) );
          xv[3] = _mm256_loadu_ps( ( float* )(x1 + 3*8 ) );
          xv[4] = _mm256_loadu_ps( ( float* )(x1 + 4*8 ) );

          yv[0] = _mm256_loadu_ps( ( float* )(y1 + 0*8 ) );
          yv[1] = _mm256_loadu_ps( ( float* )(y1 + 1*8 ) );
          yv[2] = _mm256_loadu_ps( ( float* )(y1 + 2*8 ) );
          yv[3] = _mm256_loadu_ps( ( float* )(y1 + 3*8 ) );
          yv[4] = _mm256_loadu_ps( ( float* )(y1 + 4*8 ) );

          zv[0] =  _mm256_fmadd_ps(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_ps(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_ps(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_ps(xv[3],  alpha1v, yv[3]);
          zv[4] =  _mm256_fmadd_ps(xv[4],  alpha1v, yv[4]);


          _mm256_storeu_ps( ( float* )(y1 + 0*8 ), zv[0] );
          _mm256_storeu_ps( ( float* )(y1 + 1*8 ), zv[1] );
          _mm256_storeu_ps( ( float* )(y1 + 2*8 ), zv[2] );
          _mm256_storeu_ps( ( float* )(y1 + 3*8 ), zv[3] );
          _mm256_storeu_ps( ( float* )(y1 + 4*8 ), zv[4] );

          x1 += 40;
          y1 += 40;
        }

      for (; i + 31 < n; i += 32 )
        {
          xv[0] = _mm256_loadu_ps( ( float* )(x1 + 0*8 ) );
          xv[1] = _mm256_loadu_ps( ( float* )(x1 + 1*8 ) );
          xv[2] = _mm256_loadu_ps( ( float* )(x1 + 2*8 ) );
          xv[3] = _mm256_loadu_ps( ( float* )(x1 + 3*8 ) );


          yv[0] = _mm256_loadu_ps( ( float* )(y1 + 0*8 ) );
          yv[1] = _mm256_loadu_ps( ( float* )(y1 + 1*8 ) );
          yv[2] = _mm256_loadu_ps( ( float* )(y1 + 2*8 ) );
          yv[3] = _mm256_loadu_ps( ( float* )(y1 + 3*8 ) );


          zv[0] =  _mm256_fmadd_ps(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_ps(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_ps(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_ps(xv[3],  alpha1v, yv[3]);

          _mm256_storeu_ps( ( float* )(y1 + 0*8 ), zv[0] );
          _mm256_storeu_ps( ( float* )(y1 + 1*8 ), zv[1] );
          _mm256_storeu_ps( ( float* )(y1 + 2*8 ), zv[2] );
          _mm256_storeu_ps( ( float* )(y1 + 3*8 ), zv[3] );

          x1 += 32;
          y1 += 32;
        }
      for (; i + 15 < n; i += 16 )
        {
          xv[0] = _mm256_loadu_ps( ( float* )x1 );
          xv[1] = _mm256_loadu_ps( ( float* )(x1 + 8));

          yv[0] = _mm256_loadu_ps( ( float* )y1 );
          yv[1] = _mm256_loadu_ps( ( float* )(y1 + 8 ) );

          zv[0] =  _mm256_fmadd_ps(xv[0], alpha1v, yv[0]); // x1v = alpha * x1v + y1v
          zv[1] =  _mm256_fmadd_ps(xv[1], alpha1v, yv[1]);

          _mm256_storeu_ps( ( float* )(y1), zv[0] );
          _mm256_storeu_ps( ( float* )(y1 + 8), zv[1] );

          x1 += 16;
          y1 += 16;
        }
      for (; i + 7 < n; i += 8)
        {
          xv[2] = _mm256_loadu_ps( ( float* )x1 );
          yv[2] = _mm256_loadu_ps( ( float* )y1 );

          zv[2] = _mm256_fmadd_ps(xv[2], alpha1v, yv[2]); // x1v = alpha * x1v + y1v

          _mm256_storeu_ps( ( float* )(y1), zv[2] );

          x1 += 8;
          y1 += 8;
        }

      for(; i < n; i++) {
        y[i] += (*alpha) * x[i];
      }
    }
  else
    {
      for ( i = 0; i < n; ++i )
        {
          (( *y1 )) += (( *alpha )) * (( *x1 ));

          x1 += incx;
          y1 += incy;
        }
    }
}  // End of function

void bli_daxpyv_opt_var10  (
			    conj_t           conjx,
			    dim_t            n,
			    double* restrict alpha,
			    double* restrict x, inc_t incx,
			    double* restrict y, inc_t incy,
                              cntx_t*          cntx
			    )
{
  double* restrict x1 = x;
  double* restrict y1 = y;
  dim_t  i;
  //  dim_t j;
  __m256d  alpha1v;
  __m256d xv[10];
  __m256d yv[10];
  __m256d zv[10];
  
  if ( ( (n) == 0 ) ) return;

  /* If alpha is zero, return. */
  if ( ( ((*alpha)) == (0.0F) ) ) return;


  if ( incx == 1 && incy == 1 )
    {
      alpha1v = _mm256_broadcast_sd( alpha );

      for (i = 0; (i + 39) < n; i += 40)
        {
          // 40 elements will be processed per loop. 10 FMAs will run per loop
          xv[0] = _mm256_loadu_pd( ( double* )(x1 + 0*4 ) );
          xv[1] = _mm256_loadu_pd( ( double* )(x1 + 1*4 ) );
          xv[2] = _mm256_loadu_pd( ( double* )(x1 + 2*4 ) );
          xv[3] = _mm256_loadu_pd( ( double* )(x1 + 3*4 ) );
	  xv[4] = _mm256_loadu_pd( ( double* )(x1 + 4*4 ) );
          xv[5] = _mm256_loadu_pd( ( double* )(x1 + 5*4 ) );
          xv[6] = _mm256_loadu_pd( ( double* )(x1 + 6*4 ) );
          xv[7] = _mm256_loadu_pd( ( double* )(x1 + 7*4 ) );
          xv[8] = _mm256_loadu_pd( ( double* )(x1 + 8*4 ) );
          xv[9] = _mm256_loadu_pd( ( double* )(x1 + 9*4 ) );

          yv[0] = _mm256_loadu_pd( ( double* )(y1 + 0*4 ) );
          yv[1] = _mm256_loadu_pd( ( double* )(y1 + 1*4 ) );
          yv[2] = _mm256_loadu_pd( ( double* )(y1 + 2*4 ) );
          yv[3] = _mm256_loadu_pd( ( double* )(y1 + 3*4 ) );
          yv[4] = _mm256_loadu_pd( ( double* )(y1 + 4*4 ) );
          yv[5] = _mm256_loadu_pd( ( double* )(y1 + 5*4 ) );
          yv[6] = _mm256_loadu_pd( ( double* )(y1 + 6*4 ) );
          yv[7] = _mm256_loadu_pd( ( double* )(y1 + 7*4 ) );
          yv[8] = _mm256_loadu_pd( ( double* )(y1 + 8*4 ) );
          yv[9] = _mm256_loadu_pd( ( double* )(y1 + 9*4 ) );

          zv[0] =  _mm256_fmadd_pd(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_pd(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_pd(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_pd(xv[3],  alpha1v, yv[3]);
          zv[4] =  _mm256_fmadd_pd(xv[4],  alpha1v, yv[4]);
          zv[5] =  _mm256_fmadd_pd(xv[5],  alpha1v, yv[5]);
          zv[6] =  _mm256_fmadd_pd(xv[6],  alpha1v, yv[6]);
          zv[7] =  _mm256_fmadd_pd(xv[7],  alpha1v, yv[7]);
          zv[8] =  _mm256_fmadd_pd(xv[8],  alpha1v, yv[8]);
          zv[9] =  _mm256_fmadd_pd(xv[9],  alpha1v, yv[9]);

          _mm256_storeu_pd( ( double* )(y1 + 0*4 ), zv[0] );
          _mm256_storeu_pd( ( double* )(y1 + 1*4 ), zv[1] );
          _mm256_storeu_pd( ( double* )(y1 + 2*4 ), zv[2] );
          _mm256_storeu_pd( ( double* )(y1 + 3*4 ), zv[3] );
          _mm256_storeu_pd( ( double* )(y1 + 4*4 ), zv[4] );
          _mm256_storeu_pd( ( double* )(y1 + 5*4 ), zv[5] );
          _mm256_storeu_pd( ( double* )(y1 + 6*4 ), zv[6] );
          _mm256_storeu_pd( ( double* )(y1 + 7*4 ), zv[7] );
          _mm256_storeu_pd( ( double* )(y1 + 8*4 ), zv[8] );
          _mm256_storeu_pd( ( double* )(y1 + 9*4 ), zv[9] );

          x1 += 40;
          y1 += 40;
        }
      for ( ; (i + 19) < n; i += 20 )
        {
          xv[0] = _mm256_loadu_pd( ( double* )(x1 + 0*4 ) );
          xv[1] = _mm256_loadu_pd( ( double* )(x1 + 1*4 ) );
          xv[2] = _mm256_loadu_pd( ( double* )(x1 + 2*4 ) );
          xv[3] = _mm256_loadu_pd( ( double* )(x1 + 3*4 ) );
          xv[4] = _mm256_loadu_pd( ( double* )(x1 + 4*4 ) );

          yv[0] = _mm256_loadu_pd( ( double* )(y1 + 0*4 ) );
          yv[1] = _mm256_loadu_pd( ( double* )(y1 + 1*4 ) );
          yv[2] = _mm256_loadu_pd( ( double* )(y1 + 2*4 ) );
          yv[3] = _mm256_loadu_pd( ( double* )(y1 + 3*4 ) );
          yv[4] = _mm256_loadu_pd( ( double* )(y1 + 4*4 ) );

          zv[0] =  _mm256_fmadd_pd(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_pd(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_pd(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_pd(xv[3],  alpha1v, yv[3]);
          zv[4] =  _mm256_fmadd_pd(xv[4],  alpha1v, yv[4]);


          _mm256_storeu_pd( ( double* )(y1 + 0*4 ), zv[0] );
          _mm256_storeu_pd( ( double* )(y1 + 1*4 ), zv[1] );
          _mm256_storeu_pd( ( double* )(y1 + 2*4 ), zv[2] );
          _mm256_storeu_pd( ( double* )(y1 + 3*4 ), zv[3] );
          _mm256_storeu_pd( ( double* )(y1 + 4*4 ), zv[4] );

          x1 += 20;
          y1 += 20;
        }

      for (; i + 15 < n; i += 16 )
        {
          xv[0] = _mm256_loadu_pd( ( double* )(x1 + 0*4 ) );
          xv[1] = _mm256_loadu_pd( ( double* )(x1 + 1*4 ) );
          xv[2] = _mm256_loadu_pd( ( double* )(x1 + 2*4 ) );
          xv[3] = _mm256_loadu_pd( ( double* )(x1 + 3*4 ) );


          yv[0] = _mm256_loadu_pd( ( double* )(y1 + 0*4 ) );
          yv[1] = _mm256_loadu_pd( ( double* )(y1 + 1*4 ) );
          yv[2] = _mm256_loadu_pd( ( double* )(y1 + 2*4 ) );
          yv[3] = _mm256_loadu_pd( ( double* )(y1 + 3*4 ) );


          zv[0] =  _mm256_fmadd_pd(xv[0],  alpha1v, yv[0]);
          zv[1] =  _mm256_fmadd_pd(xv[1],  alpha1v, yv[1]);
          zv[2] =  _mm256_fmadd_pd(xv[2],  alpha1v, yv[2]);
          zv[3] =  _mm256_fmadd_pd(xv[3],  alpha1v, yv[3]);

          _mm256_storeu_pd( ( double* )(y1 + 0*4 ), zv[0] );
          _mm256_storeu_pd( ( double* )(y1 + 1*4 ), zv[1] );
          _mm256_storeu_pd( ( double* )(y1 + 2*4 ), zv[2] );
          _mm256_storeu_pd( ( double* )(y1 + 3*4 ), zv[3] );

          x1 += 16;
          y1 += 16;
        }
      for (; i + 7 < n; i += 8 )
        {
          xv[0] = _mm256_loadu_pd( ( double* )x1 );
          xv[1] = _mm256_loadu_pd( ( double* )(x1 + 4));

          yv[0] = _mm256_loadu_pd( ( double* )y1 );
          yv[1] = _mm256_loadu_pd( ( double* )(y1 + 4 ) );

          zv[0] =  _mm256_fmadd_pd(xv[0], alpha1v, yv[0]); // x1v = alpha * x1v + y1v
          zv[1] =  _mm256_fmadd_pd(xv[1], alpha1v, yv[1]);

          _mm256_storeu_pd( ( double* )(y1), zv[0] );
          _mm256_storeu_pd( ( double* )(y1 + 4), zv[1] );

          x1 += 8;
          y1 += 8;
        }
      for (; i + 3 < n; i += 4)
        {
          xv[2] = _mm256_loadu_pd( ( double* )x1 );
          yv[2] = _mm256_loadu_pd( ( double* )y1 );

          zv[2] = _mm256_fmadd_pd(xv[2], alpha1v, yv[2]); // x1v = alpha * x1v + y1v

          _mm256_storeu_pd( ( double* )(y1), zv[2] );

          x1 += 4;
          y1 += 4;
        }

      for(; i < n; i++) {
        y[i] += (*alpha) * x[i];
      }
    }
  else
    {
      for ( i = 0; i < n; ++i )
        {
          (( *y1 )) += (( *alpha )) * (( *x1 ));

          x1 += incx;
          y1 += incy;
        }
    }
}  // End of function
