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
*  AVX 256 bit register holds 8 SP data*/
typedef union
{
    __m256 v;
    float f[8];
}v8ff_t;

/* ! /brief Single precision dotxf function.
*
*  Input:
*       conja : ignore for real data types
*       conjx : ignore for real data types
*       m     : input vector length
*       b_n   : fuse factor
*       alpha : multiplier
*       a     : pointer to matrix
*       inca  : matrix increment
*       lda   : matrix stride
*       x     : pointer to vector
*       incx  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       y     : pointer to vector
*       incy  : vector increment
*/
void bli_sdotxf_int_var1
     (
       conj_t          conjat,
       conj_t          conjx,
       dim_t           m,
       dim_t           b_n,
       float* restrict alpha,
       float* restrict a, inc_t inca, inc_t lda,
       float* restrict x, inc_t incx,
       float* restrict beta,
       float* restrict y, inc_t incy,
       cntx_t*         cntx
     )
{

    dim_t  i, j;
    bool_t use_ref = FALSE;
    const dim_t fusefac = 8;
    const dim_t n_elem_per_reg = 8;

    float*  restrict a0;
    float*  restrict a1;
    float*  restrict a2;
    float*  restrict a3;
    float*  restrict a4;
    float*  restrict a5;
    float*  restrict a6;
    float*  restrict a7;

    float*  restrict y0;
    float*  restrict y1;
    float*  restrict y2;
    float*  restrict y3;
    float*  restrict y4;
    float*  restrict y5;
    float*  restrict y6;
    float*  restrict y7;

    float*  restrict x0;

    float   rho0, rho1, rho2, rho3;
    float   rho4, rho5, rho6, rho7;

    float   a0c, a1c, a2c, a3c, x0c;
    float   a4c, a5c, a6c, a7c;

    dim_t m_run, m_left;
    v8ff_t a0c_vec, a1c_vec, a2c_vec, a3c_vec;
    v8ff_t a4c_vec, a5c_vec, a6c_vec, a7c_vec;
    v8ff_t x0c_vec, rho0_vec, rho1_vec, rho2_vec, rho3_vec;
    v8ff_t rho4_vec, rho5_vec, rho6_vec, rho7_vec;

    if ( bli_zero_dim1( b_n ) ) return;

    // If the vector lengths are zero, scale r by beta and return.
    if ( bli_zero_dim1( m ) )
    {
        bli_sscalv( BLIS_NO_CONJUGATE,
                    b_n,
                    beta,
                    y, incy,
                    cntx );
        return;
    }

    if ( b_n > fusefac )
    {
        use_ref = TRUE;
    }
    else if ( inca != 1 || incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }
    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {

        BLIS_SDOTXF_KERNEL_REF( conjat,
                                conjx,
                                m,
                                b_n,
                                alpha,
                                a, inca, lda,
                                x, incx,
                                beta,
                                y, incy,
                                cntx );
        return;
    }


    m_run =  m / n_elem_per_reg;
    m_left = m % n_elem_per_reg;

    rho0_vec.v = _mm256_setzero_ps();
    rho1_vec.v = _mm256_setzero_ps();
    rho2_vec.v = _mm256_setzero_ps();
    rho3_vec.v = _mm256_setzero_ps();
    rho4_vec.v = _mm256_setzero_ps();
    rho5_vec.v = _mm256_setzero_ps();
    rho6_vec.v = _mm256_setzero_ps();
    rho7_vec.v = _mm256_setzero_ps();

    if(b_n == fusefac)
    {
        a0 = a;
        a1 = a +   lda;
        a2 = a + 2*lda;
        a3 = a + 3*lda;
        a4 = a + 4*lda;
        a5 = a + 5*lda;
        a6 = a + 6*lda;
        a7 = a + 7*lda;

        x0 = x;

        for(i = 0; i < m_run; i++)
        {

            // load the input
            a0c_vec.v = _mm256_loadu_ps(a0);
            a1c_vec.v = _mm256_loadu_ps(a1);
            a2c_vec.v = _mm256_loadu_ps(a2);
            a3c_vec.v = _mm256_loadu_ps(a3);
            a4c_vec.v = _mm256_loadu_ps(a4);
            a5c_vec.v = _mm256_loadu_ps(a5);
            a6c_vec.v = _mm256_loadu_ps(a6);
            a7c_vec.v = _mm256_loadu_ps(a7);

            x0c_vec.v = _mm256_loadu_ps(x0);

            // Calculate the dot product
            rho0_vec.v += a0c_vec.v * x0c_vec.v;
            rho1_vec.v += a1c_vec.v * x0c_vec.v;
            rho2_vec.v += a2c_vec.v * x0c_vec.v;
            rho3_vec.v += a3c_vec.v * x0c_vec.v;
            rho4_vec.v += a4c_vec.v * x0c_vec.v;
            rho5_vec.v += a5c_vec.v * x0c_vec.v;
            rho6_vec.v += a6c_vec.v * x0c_vec.v;
            rho7_vec.v += a7c_vec.v * x0c_vec.v;

            a0 += n_elem_per_reg;
            a1 += n_elem_per_reg;
            a2 += n_elem_per_reg;
            a3 += n_elem_per_reg;
            a4 += n_elem_per_reg;
            a5 += n_elem_per_reg;
            a6 += n_elem_per_reg;
            a7 += n_elem_per_reg;
            x0 += n_elem_per_reg;

        }

        // Accumulate the output from vector register
        rho0 = rho0_vec.f[0] + rho0_vec.f[1] + rho0_vec.f[2] + rho0_vec.f[3] +
           rho0_vec.f[4] + rho0_vec.f[5] + rho0_vec.f[6] + rho0_vec.f[7];
        rho1 = rho1_vec.f[0] + rho1_vec.f[1] + rho1_vec.f[2] + rho1_vec.f[3] +
           rho1_vec.f[4] + rho1_vec.f[5] + rho1_vec.f[6] + rho1_vec.f[7];
        rho2 = rho2_vec.f[0] + rho2_vec.f[1] + rho2_vec.f[2] + rho2_vec.f[3] +
           rho2_vec.f[4] + rho2_vec.f[5] + rho2_vec.f[6] + rho2_vec.f[7];
        rho3 = rho3_vec.f[0] + rho3_vec.f[1] + rho3_vec.f[2] + rho3_vec.f[3] +
           rho3_vec.f[4] + rho3_vec.f[5] + rho3_vec.f[6] + rho3_vec.f[7];


        rho4 = rho4_vec.f[0] + rho4_vec.f[1] + rho4_vec.f[2] + rho4_vec.f[3] +
           rho4_vec.f[4] + rho4_vec.f[5] + rho4_vec.f[6] + rho4_vec.f[7];
        rho5 = rho5_vec.f[0] + rho5_vec.f[1] + rho5_vec.f[2] + rho5_vec.f[3] +
           rho5_vec.f[4] + rho5_vec.f[5] + rho5_vec.f[6] + rho5_vec.f[7];
        rho6 = rho6_vec.f[0] + rho6_vec.f[1] + rho6_vec.f[2] + rho6_vec.f[3] +
           rho6_vec.f[4] + rho6_vec.f[5] + rho6_vec.f[6] + rho6_vec.f[7];
        rho7 = rho7_vec.f[0] + rho7_vec.f[1] + rho7_vec.f[2] + rho7_vec.f[3] +
           rho7_vec.f[4] + rho7_vec.f[5] + rho7_vec.f[6] + rho7_vec.f[7];


        // if input data size is non multiple of the number of elements in vector register
        for(i = 0; i < m_left; i++)
        {
            a0c = *a0;
            a1c = *a1;
            a2c = *a2;
            a3c = *a3;
            a4c = *a4;
            a5c = *a5;
            a6c = *a6;
            a7c = *a7;

            x0c = *x0;

            rho0 += a0c * x0c;
            rho1 += a1c * x0c;
            rho2 += a2c * x0c;
            rho3 += a3c * x0c;
            rho4 += a4c * x0c;
            rho5 += a5c * x0c;
            rho6 += a6c * x0c;
            rho7 += a7c * x0c;

            a0 += 1;
            a1 += 1;
            a2 += 1;
            a3 += 1;
            a4 += 1;
            a5 += 1;
            a6 += 1;
            a7 += 1;
            x0 += 1;
        }

        y0 = y;
        y1 = y0 + 1;
        y2 = y1 + 1;
        y3 = y2 + 1;
        y4 = y3 + 1;
        y5 = y4 + 1;
        y6 = y5 + 1;
        y7 = y6 + 1;
        
        //store the output data
        (*y0) = (*y0) * (*beta) + rho0 * (*alpha);
        (*y1) = (*y1) * (*beta) + rho1 * (*alpha);
        (*y2) = (*y2) * (*beta) + rho2 * (*alpha);
        (*y3) = (*y3) * (*beta) + rho3 * (*alpha);
        (*y4) = (*y4) * (*beta) + rho4 * (*alpha);
        (*y5) = (*y5) * (*beta) + rho5 * (*alpha);
        (*y6) = (*y6) * (*beta) + rho6 * (*alpha);
        (*y7) = (*y7) * (*beta) + rho7 * (*alpha);
    }
    else
    {
        // the case where b_n is less than fuse factor       
        for ( i = 0; i < b_n; ++i )
        {
            a0   = a + (0  )*inca + (i  )*lda;
            x0   = x + (0  )*incx;
            y0   = y + (i  )*incy;
            rho0 = 0.0;

            m_run =  m / n_elem_per_reg;
            m_left = m % n_elem_per_reg;

            rho0_vec.v = _mm256_setzero_ps();

            for(j = 0; j < m_run; j++)
            {
                a0c_vec.v = _mm256_loadu_ps(a0);
                x0c_vec.v = _mm256_loadu_ps(x0);
                rho0_vec.v +=  x0c_vec.v *  a0c_vec.v;

                a0 += n_elem_per_reg;
                x0 += n_elem_per_reg;
            }

           rho0 = rho0_vec.f[0] + rho0_vec.f[1] + rho0_vec.f[2] + rho0_vec.f[3] +
                       rho0_vec.f[4] + rho0_vec.f[5] + rho0_vec.f[6] + rho0_vec.f[7];

           for(j = 0; j < m_left; j++)
           {
                rho0 += a0[j] * x0[j];
           }

           (*y0) = (*y0) *  (*beta) + rho0 * (*alpha);
        }
    }
}


/* Union data structure to access AVX registers
*  AVX 256 bit register holds 4 DP data*/
typedef union
{
    __m256d v;
    double d[4];
}v4df_t;

/* ! /brief Double precision dotxf function.
*
*  Input:
*       conjat: ignore for real data types
*       conjx : ignore for real data types
*       m     : input vector length
*       b_n   : fuse factor
*       alpha : multiplier
*       a     : pointer to matrix
*       inca  : matrix increment
*       lda   : matrix stride
*       x     : pointer to vector
*       incx  : vector increment
*       cntx  : BLIS context pointer
*  Output:
*       y     : pointer to vector
*       incy  : vector increment
*/
void bli_ddotxf_int_var1
     (
       conj_t          conjat,
       conj_t          conjx,
       dim_t           m,
       dim_t           b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t*         cntx
     )
{

    double*  restrict alpha_cast = alpha;
    double*  restrict beta_cast = beta;
    double*  restrict a_cast = a;
    double*  restrict x_cast = x;
    double*  restrict y_cast = y;

    double*  restrict a0;
    double*  restrict a1;
    double*  restrict a2;
    double*  restrict a3;

    double*  restrict y0;
    double*  restrict y1;
    double*  restrict y2;
    double*  restrict y3;

    double*  restrict x0;

    double            rho0, rho1, rho2, rho3;
    double            a0c, a1c, a2c, a3c, x0c;
    dim_t  i;
    bool_t            use_ref = FALSE;
    const dim_t fusefac = 4;
    const dim_t n_elem_per_reg = 4;

    dim_t m_run, m_left;
    v4df_t a0c_vec, a1c_vec, a2c_vec, a3c_vec;
    v4df_t rho0_vec, rho1_vec, rho2_vec, rho3_vec;
    v4df_t x0c_vec;

    if ( bli_zero_dim1( b_n ) ) return;

    // If the vector lengths are zero, scale r by beta and return.
    if ( bli_zero_dim1( m ) )
    {
        bli_dscalv( BLIS_NO_CONJUGATE,
                    b_n,
                    beta_cast,
                    y_cast, incy,
                    cntx );
        return;
    }

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
    if ( b_n != fusefac )
    {
        use_ref = TRUE;
    }
    else if ( inca != 1 || incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

    // Call the reference implementation if needed.
    if ( use_ref == TRUE )
    {

        BLIS_DDOTXF_KERNEL_REF( conjat,
                                conjx,
                                m,
                                b_n,
                                alpha_cast,
                                a_cast, inca, lda,
                                x_cast, incx,
                                beta_cast,
                                y_cast, incy,
                                cntx );
        return;
    }

    a0 = a_cast;
    a1 = a_cast +   lda;
    a2 = a_cast + 2*lda;
    a3 = a_cast + 3*lda;
    x0 = x_cast;

    m_run =  m / n_elem_per_reg;
    m_left = m % n_elem_per_reg;

    rho0_vec.v = _mm256_setzero_pd();
    rho1_vec.v = _mm256_setzero_pd();
    rho2_vec.v = _mm256_setzero_pd();
    rho3_vec.v = _mm256_setzero_pd();

    for(i = 0; i < m_run; i++)
    {
        // load the input
        a0c_vec.v = _mm256_loadu_pd(a0);
        a1c_vec.v = _mm256_loadu_pd(a1);
        a2c_vec.v = _mm256_loadu_pd(a2);
        a3c_vec.v = _mm256_loadu_pd(a3);

        x0c_vec.v = _mm256_loadu_pd(x0);

        //calculate the dot product
        rho0_vec.v += a0c_vec.v * x0c_vec.v;
        rho1_vec.v += a1c_vec.v * x0c_vec.v;
        rho2_vec.v += a2c_vec.v * x0c_vec.v;
        rho3_vec.v += a3c_vec.v * x0c_vec.v;

        a0 += n_elem_per_reg;
        a1 += n_elem_per_reg;
        a2 += n_elem_per_reg;
        a3 += n_elem_per_reg;
        x0 += n_elem_per_reg;

    }

    // Accumulate the output from vector register
    rho0 = rho0_vec.d[0] + rho0_vec.d[1] + rho0_vec.d[2] + rho0_vec.d[3];
    rho1 = rho1_vec.d[0] + rho1_vec.d[1] + rho1_vec.d[2] + rho1_vec.d[3];
    rho2 = rho2_vec.d[0] + rho2_vec.d[1] + rho2_vec.d[2] + rho2_vec.d[3];
    rho3 = rho3_vec.d[0] + rho3_vec.d[1] + rho3_vec.d[2] + rho3_vec.d[3];

    // if input data size is non multiple of the number of elements in vector register
    for(i = 0; i < m_left; i++)
    {

        a0c = *a0;
        a1c = *a1;
        a2c = *a2;
        a3c = *a3;
        x0c = *x0;

        rho0 += a0c * x0c;
        rho1 += a1c * x0c;
        rho2 += a2c * x0c;
        rho3 += a3c * x0c;

        a0 += 1;
        a1 += 1;
        a2 += 1;
        a3 += 1;
        x0 += 1;
    }

    y0 = y_cast;
    y1 = y0 + 1;
    y2 = y1 + 1;
    y3 = y2 + 1;

    //store the output data
    (*y0) = (*y0) * (*beta_cast) + rho0 * (*alpha_cast);
    (*y1) = (*y1) * (*beta_cast) + rho1 * (*alpha_cast);
    (*y2) = (*y2) * (*beta_cast) + rho2 * (*alpha_cast);
    (*y3) = (*y3) * (*beta_cast) + rho3 * (*alpha_cast);
}
