/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"
#include "blis.h"

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
  __m256d v;
  double  d[4] __attribute__((aligned(64)));
} v4df_t;

/**
 * GEMV Operation (assuming op(A) = TRANSPOSE):
 *      y := beta * y + alpha * op(A) * x
 *  where,
 *      y - n-dimensional vector when op(A) = TRANSPOSE.
 *      x - m-dimensional vector when op(A) = TRANSPOSE.
 *      A - m x n dimensional matrix.
 *      alpha, beta - scalars.
 */

// Function pointers for n-biased kernels.
static dgemv_ker_ft n_ker_fp[] =
{
    NULL,
    bli_dgemv_t_zen_int_mx1_avx2,   // n = 1
    bli_dgemv_t_zen_int_mx2_avx2,   // n = 2
    bli_dgemv_t_zen_int_mx3_avx2,   // n = 3
    bli_dgemv_t_zen_int_mx4_avx2,   // n = 4
    bli_dgemv_t_zen_int_mx5_avx2,   // n = 5
    bli_dgemv_t_zen_int_mx6_avx2,   // n = 6
    bli_dgemv_t_zen_int_mx7_avx2    // n = 7
};

/**
 * bli_dgemv_t_zen_int_avx2(...) handles cases where op(A) = TRANSPOSE && column-storage
 * or op(A) = NON-TRANSPOSE && row-storage. We will compute 8 columns of A at a time until
 * less than 8 columns remain. Then we will then call others kernels to handle fringe cases.
 *
 * Here we will use dot product to multiply each column of matrix A with vector x in
 * groups of 8 columns, resulting product will be stored in a temporary vector before
 * being added to vector y.
 *
 * In case of non-transpose row storage, the values of m, n, inca and lda will be interchanged
 *  by bli_dgemv_unf_var1(...)
 *
 *                         i=0     i=1      ...
 *      |---|           |-------|-------|---------|          |---|
 *      |   |           |       |       |         |          |   |
 *  i=0 | 8 |           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *      |---|           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *  i=1 | 8 |   :=      |       |       |         |   *      |   |
 *      |   |           |       |       |         |          |   |
 *      |---|         j | m x 8 | m x 8 |         |        j | m |
 *      |   |           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *  ... |   |           |       |       |         |      ... |   |
 *      |   |           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *      |   |           |       |       |         |          |   |
 *      |---|           |-------|-------|---------|          |---|
 *
 *        y                         A                        x
 *     (n x 1)                   (m x n)                  (m x 1)
 *                             column storage
 */
void bli_dgemv_t_zen_int_avx2
     (
       conj_t  conja,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double* alpha,
       double* a, inc_t inca, inc_t lda,
       double* x, inc_t incx,
       double* beta,
       double* y, inc_t incy,
       cntx_t* cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of rows completed
    dim_t i = 0;

    // n_iter is the number of times the loop needs to repeat
    dim_t n_iter = n / 8;
    dim_t n_left = n % 8;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3};

    // n_elem_per_reg is 4 since we are using avx2
    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask  = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // vector variables used load inputs.
    v4df_t yv0, yv1;                                                        // yv0 --> y_buf[0:3], yv1 --> y_buf[4:7]
    v4df_t xv0, xv1, xv2, xv3;                                              // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[8];                                                         // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] & rhov[1] --> Result for (a_buf[:, 0:7] * x[:])
    v4df_t a_vec[8];                                                        // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 8 rows of A (columns of A^T).
    double *restrict av[8];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );


    // The following loop processes the matrix A in chunks of 8 columns at a time.
    // For each chunk, the result of the matrix-vector multiplication is stored in the corresponding elements of the vector y.
    for ( i = 0; i < n_iter; ++i)
    {
        // j denotes the number of rows completed
        dim_t j = 0;

        // Creating an array of pointers for 8 columns of matrix A
        av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[:, 0]
        av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[:, 1]
        av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[:, 2]
        av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[:, 3]

        av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[:, 4]
        av[5] = a_buf + 5 * lda;                                           // av[5] = a_buf[:, 5]
        av[6] = a_buf + 6 * lda;                                           // av[6] = a_buf[:, 6]
        av[7] = a_buf + 7 * lda;                                           // av[7] = a_buf[:, 7]

        // Clearing vectors for next loop
        rhov[0].v = _mm256_setzero_pd();
        rhov[1].v = _mm256_setzero_pd();
        rhov[2].v = _mm256_setzero_pd();
        rhov[3].v = _mm256_setzero_pd();

        rhov[4].v = _mm256_setzero_pd();
        rhov[5].v = _mm256_setzero_pd();
        rhov[6].v = _mm256_setzero_pd();
        rhov[7].v = _mm256_setzero_pd();

        // Loading data from vector y
        if (bli_deq0( *beta ))
        {
            // yv0 and yv1 are assigned zero if beta is 0
            yv0.v = _mm256_setzero_pd();
            yv1.v = _mm256_setzero_pd();
        }
        else if ( incy == 1)
        {
            // In case of unit stride y, the inputs on vector are moved to register yv using vector load
            // Followed by '_mm5256_mul_pd' to get beta * y
            yv0.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf ) );        // yv0 = beta * y_buf[0:3]
            yv1.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf + 4 ) );    // yv1 = beta * y_buf[4:7]
        }
        else
        {
            // In case of non-unit stride y,
            // The inputs on vector y are manually moved to registor yv0
            yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
            yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
            yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]
            yv0.d[3] = *( y_buf + 3 * incy);                                // yv0[3] = y_buf[3]

            yv1.d[0] = *( y_buf + 4 * incy);                                // yv1[0] = y_buf[4]
            yv1.d[1] = *( y_buf + 5 * incy);                                // yv1[1] = y_buf[5]
            yv1.d[2] = *( y_buf + 6 * incy);                                // yv1[2] = y_buf[6]
            yv1.d[3] = *( y_buf + 7 * incy);                                // yv1[3] = y_buf[7]

            yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:3]
            yv1.v = _mm256_mul_pd( betav.v, yv1.v );                        // yv1 = beta * yv1 = beta * y_buf[4:7]
        }

        // Handles (x_buf[0:15] * a_buf[0:15,0:7])
        for ( j = 0; (j + 15) < m; j += 16 )
        {
            // Load the input values from vector X.
            xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:3, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:3, 6] * x_buf[0:3]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:3, 7] * x_buf[0:3]

            // Load the input values from vector X.
            xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] + 4 );                     // a_vec[6] = a_buf[4:7, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] + 4 );                     // a_vec[7] = a_buf[4:7, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[4:7, 6] * x_buf[4:7]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv1.v, rhov[7].v );    // rhov[7] += a_buf[4:7, 7] * x_buf[4:7]

            // Load the input values from vector X.
            xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:11, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[8:11, 3] * x_buf[8:11]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:11, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:11, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:11, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] + 8 );                     // a_vec[7] = a_buf[8:11, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[8:11, 4] * x_buf[8:11]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[8:11, 5] * x_buf[8:11]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv2.v, rhov[6].v );    // rhov[6] += a_buf[8:11, 6] * x_buf[8:11]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv2.v, rhov[7].v );    // rhov[7] += a_buf[8:11, 7] * x_buf[8:11]

            // Load the input values from vector X.
            xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] + 12 );                    // a_vec[3] = a_buf[12:15, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 3] * x_buf[12:15]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] + 12 );                    // a_vec[4] = a_buf[12:15, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] + 12 );                    // a_vec[5] = a_buf[12:15, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] + 12 );                    // a_vec[6] = a_buf[12:15, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] + 12 );                    // a_vec[7] = a_buf[12:15, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[12:15, 4] * x_buf[12:15]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[12:15, 5] * x_buf[12:15]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv3.v, rhov[6].v );    // rhov[6] += a_buf[12:15, 6] * x_buf[12:15]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv3.v, rhov[7].v );    // rhov[7] += a_buf[12:15, 7] * x_buf[12:15]

            // Incrementing pointers by 16 (4 iterations * 4 elements per register)
            av[0] += 16;
            av[1] += 16;
            av[2] += 16;
            av[3] += 16;
            av[4] += 16;
            av[5] += 16;
            av[6] += 16;
            av[7] += 16;
            x_buf += 16;
        }

        // Handles (x_buf[0:7] * a_buf[0:7,0:7])
        if ( (j + 7) < m )
        {
            // Load the input values from vector X.
            xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:3, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:3, 6] * x_buf[0:3]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:3, 7] * x_buf[0:3]

            // Load the input values from vector X.
            xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] + 4 );                     // a_vec[6] = a_buf[4:7, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] + 4 );                     // a_vec[7] = a_buf[4:7, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[4:7, 6] * x_buf[4:7]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv1.v, rhov[7].v );    // rhov[7] += a_buf[4:7, 7] * x_buf[4:7]

            // Incrementing pointers by 8 (2 iterations * 4 elements per register)
            av[0] += 8;
            av[1] += 8;
            av[2] += 8;
            av[3] += 8;
            av[4] += 8;
            av[5] += 8;
            av[6] += 8;
            av[7] += 8;
            x_buf += 8;
            j     += 8;
        }

        // Handles (x_buf[0:3] * a_buf[0:7,0:3])
        if ( (j + 3) < m )
        {
            // Load the input values from vector X.
            xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
            a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
            a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
            a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
            a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
            a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]
            a_vec[7].v =  _mm256_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:3, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[4, 0:7] * x_buf[0:3]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[5, 0:7] * x_buf[0:3]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[6, 0:7] * x_buf[0:3]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[7, 0:7] * x_buf[0:3]

            // Incrementing pointers by 4 (1 iteration * 4 elements per register)
            av[0] += 4;
            av[1] += 4;
            av[2] += 4;
            av[3] += 4;
            av[4] += 4;
            av[5] += 4;
            av[6] += 4;
            av[7] += 4;
            x_buf += 4;
            j    += 4;
        }

        // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left, 0:7])
        if( m_left )
        {
            // Load the input values from vector X.
            xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

            // Load the input values from Matrix A
            a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
            a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
            a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]
            a_vec[3].v = _mm256_maskload_pd(av[3], m_mask);                 // a_vec[3] = a_buf[0:m_left, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
            rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
            rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]
            rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:3]

            // Load the input values from Matrix A
            a_vec[4].v = _mm256_maskload_pd(av[4], m_mask);                 // a_vec[4] = a_buf[0:m_left, 4]
            a_vec[5].v = _mm256_maskload_pd(av[5], m_mask);                 // a_vec[5] = a_buf[0:m_left, 5]
            a_vec[6].v = _mm256_maskload_pd(av[6], m_mask);                 // a_vec[6] = a_buf[0:m_left, 6]
            a_vec[7].v = _mm256_maskload_pd(av[7], m_mask);                 // a_vec[7] = a_buf[0:m_left, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:3]
            rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:3]
            rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:m_left, 6] * x_buf[0:3]
            rhov[7].v = _mm256_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:m_left, 7] * x_buf[0:3]
        }
        // This section of code is used to find the sum of values in 8 vectors (rhov[0:7]),
        // and store the result into the 4 elements of rhov[0] and rhov[1] vectors each.
        rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                  //rhov[0][0:1] = rhov[0][0] + rhov[0][1], rhov[0][2:3] = rhov[0][2] + rhov[0][3]
        rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                  //rhov[1][0:1] = rhov[1][0] + rhov[1][1], rhov[1][2:3] = rhov[1][2] + rhov[1][3]
        rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                  //rhov[2][0:1] = rhov[2][0] + rhov[2][1], rhov[2][2:3] = rhov[2][2] + rhov[2][3]
        rhov[3].v = _mm256_hadd_pd( rhov[3].v, rhov[3].v );                  //rhov[3][0:1] = rhov[3][0] + rhov[3][1], rhov[3][2:3] = rhov[3][2] + rhov[3][3]

        rhov[4].v = _mm256_hadd_pd( rhov[4].v, rhov[4].v );                  //rhov[4][0:1] = rhov[4][0] + rhov[4][1], rhov[4][2:3] = rhov[4][2] + rhov[4][3]
        rhov[5].v = _mm256_hadd_pd( rhov[5].v, rhov[5].v );                  //rhov[5][0:1] = rhov[5][0] + rhov[5][1], rhov[5][2:3] = rhov[5][2] + rhov[5][3]
        rhov[6].v = _mm256_hadd_pd( rhov[6].v, rhov[6].v );                  //rhov[6][0:1] = rhov[6][0] + rhov[6][1], rhov[6][2:3] = rhov[6][2] + rhov[6][3]
        rhov[7].v = _mm256_hadd_pd( rhov[7].v, rhov[7].v );                  //rhov[7][0:1] = rhov[7][0] + rhov[7][1], rhov[7][2:3] = rhov[7][2] + rhov[7][3]

        // Sum the results of the horizontal adds to get the final sums for each vector
        rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
        rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
        rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])
        rhov[0].d[3] = rhov[3].d[0] + rhov[3].d[2];                         // rhov[0][3] = (rhov[3][0] + rhov[3][1]) + (rhov[3][2] + rhov[3][3])

        // yv0 = alpha * rho + [beta * y_buf[0:3]]
        yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

        // Sum the results of the horizontal adds to get the final sums for each vector
        rhov[1].d[0] = rhov[4].d[0] + rhov[4].d[2];                         // rhov[1][0] = (rhov[4][0] + rhov[4][1]) + (rhov[4][2] + rhov[4][3])
        rhov[1].d[1] = rhov[5].d[0] + rhov[5].d[2];                         // rhov[1][1] = (rhov[5][0] + rhov[5][1]) + (rhov[5][2] + rhov[5][3])
        rhov[1].d[2] = rhov[6].d[0] + rhov[6].d[2];                         // rhov[1][2] = (rhov[6][0] + rhov[6][1]) + (rhov[6][2] + rhov[6][3])
        rhov[1].d[3] = rhov[7].d[0] + rhov[7].d[2];                         // rhov[1][3] = (rhov[7][0] + rhov[7][1]) + (rhov[7][2] + rhov[7][3])

        // yv1 = alpha * rho + [beta * y_buf[4:7]]
        yv1.v = _mm256_fmadd_pd( alphav.v, rhov[1].v, yv1.v );

        // Store the result back into vector y
        if ( incy == 1)
        {
            // In case of unit stride y,
            // The result is moved back onto vector y using '_mm256_storeu_pd'
            _mm256_storeu_pd( y_buf, yv0.v );                               // y_buf[0:3] = yv0
            _mm256_storeu_pd( y_buf + 4, yv1.v );                           // y_buf[4:7] = yv1
        }
        else
        {
            // In case of non-unit stride y,
            // The result is manually moved back onto vector y
            *( y_buf + 0 * incy ) = yv0.d[0];
            *( y_buf + 1 * incy ) = yv0.d[1];
            *( y_buf + 2 * incy ) = yv0.d[2];
            *( y_buf + 3 * incy ) = yv0.d[3];

            *( y_buf + 4 * incy) = yv1.d[0];
            *( y_buf + 5 * incy) = yv1.d[1];
            *( y_buf + 6 * incy) = yv1.d[2];
            *( y_buf + 7 * incy) = yv1.d[3];
        }

        // The pointers are moved to the corresponding position for next calculation.
        x_buf = x;
        y_buf += 8 * incy;
        a_buf += 8 * lda;
    }

    // The fringe rows are calculated in this code section.
    if ( n_left != 0 )
    {
        n_ker_fp[n_left]
        (
            conja,
            conjx,
            m,
            n_left,
            alpha,
            a_buf, inca, lda,
            x_buf, incx,
            beta,
            y_buf, incy,
            cntx
        );
    }
    return;
}

void  bli_dgemv_t_zen_int_mx7_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // Mask used to load elements 4 + 3 in vector y.
    __m256i n_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[3]);

    // vector variables used load inputs.
    v4df_t yv0, yv1;                                                   // yv0, yv1  -->   Stores 7 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[7];                                                    // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] & rhov[1] --> Result for (a_buf[:, 0:6] * x[:])
    v4df_t a_vec[7];                                                   // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 7 rows of A (columns of A^T).
    double *restrict av[7];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[2,:]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[3,:]

    av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[4,:]
    av[5] = a_buf + 5 * lda;                                           // av[5] = a_buf[5,:]
    av[6] = a_buf + 6 * lda;                                           // av[6] = a_buf[6,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();

    rhov[4].v = _mm256_setzero_pd();
    rhov[5].v = _mm256_setzero_pd();
    rhov[6].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 and yv1 are assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
        yv1.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf ) );                    // yv0 = beta * y_buf[0:3]
        yv1.v =  _mm256_mul_pd( betav.v, _mm256_maskload_pd( y_buf + 4, n_mask ) );     // yv1 = beta * y_buf[4:6]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy);                                // yv0[3] = y_buf[3]

        yv1.d[0] = *( y_buf + 4 * incy);                                // yv1[0] = y_buf[4]
        yv1.d[1] = *( y_buf + 5 * incy);                                // yv1[1] = y_buf[5]
        yv1.d[2] = *( y_buf + 6 * incy);                                // yv1[2] = y_buf[6]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:3]
        yv1.v =  _mm256_mul_pd( betav.v, yv1.v );                       // yv1 = beta * yv1 = beta * y_buf[4:6]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0:6])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:3, 6] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] + 4 );                     // a_vec[6] = a_buf[4:7, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[4:7, 6] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:11, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[8:11, 3] * x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:11, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:11, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:11, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[8:11, 4] * x_buf[8:11]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[8:11, 5] * x_buf[8:11]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv2.v, rhov[6].v );    // rhov[6] += a_buf[8:11, 6] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 12 );                    // a_vec[3] = a_buf[12:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 3] * x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 12 );                    // a_vec[4] = a_buf[12:15, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 12 );                    // a_vec[5] = a_buf[12:15, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] + 12 );                    // a_vec[6] = a_buf[12:15, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[12:15, 4] * x_buf[12:15]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[12:15, 5] * x_buf[12:15]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv3.v, rhov[6].v );    // rhov[6] += a_buf[12:15, 6] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        av[5] += 16;
        av[6] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:6])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:3, 6] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] + 4 );                     // a_vec[6] = a_buf[4:7, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[4:7, 6] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;
        av[4] += 8;
        av[5] += 8;
        av[6] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles (x_buf[0:3] * a_buf[0:3,0:6])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]
        a_vec[6].v =  _mm256_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:3, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[4, 0:7] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[5, 0:7] * x_buf[0:3]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[6, 0:7] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        av[2] += 4;
        av[3] += 4;
        av[4] += 4;
        av[5] += 4;
        av[6] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:6])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v = _mm256_maskload_pd(av[3], m_mask);                 // a_vec[3] = a_buf[0:m_left, 3]


        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[4].v = _mm256_maskload_pd(av[4], m_mask);                 // a_vec[4] = a_buf[0:m_left, 4]
        a_vec[5].v = _mm256_maskload_pd(av[5], m_mask);                 // a_vec[5] = a_buf[0:m_left, 5]
        a_vec[6].v = _mm256_maskload_pd(av[6], m_mask);                 // a_vec[6] = a_buf[0:m_left, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:3]
        rhov[6].v = _mm256_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:m_left, 6] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 7 vectors (rhov[0:6]),
    // and store the result into the 4 elements of rhov[0] and rhov[1] vectors each.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]
    rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                 // rhov[2][0:1] = rhov[2][0] + rhov[2][1]  &  rhov[2][2:3] = rhov[2][2] + rhov[2][3]
    rhov[3].v = _mm256_hadd_pd( rhov[3].v, rhov[3].v );                 // rhov[3][0:1] = rhov[3][0] + rhov[3][1]  &  rhov[3][2:3] = rhov[3][2] + rhov[3][3]

    rhov[4].v = _mm256_hadd_pd( rhov[4].v, rhov[4].v );                 // rhov[4][0:1] = rhov[4][0] + rhov[4][1]  &  rhov[4][2:3] = rhov[4][2] + rhov[4][3]
    rhov[5].v = _mm256_hadd_pd( rhov[5].v, rhov[5].v );                 // rhov[5][0:1] = rhov[5][0] + rhov[5][1]  &  rhov[5][2:3] = rhov[5][2] + rhov[5][3]
    rhov[6].v = _mm256_hadd_pd( rhov[6].v, rhov[6].v );                 // rhov[6][0:1] = rhov[6][0] + rhov[6][1]  &  rhov[6][2:3] = rhov[6][2] + rhov[6][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
    rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])
    rhov[0].d[3] = rhov[3].d[0] + rhov[3].d[2];                         // rhov[0][3] = (rhov[3][0] + rhov[3][1]) + (rhov[3][2] + rhov[3][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:3]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[1].d[0] = rhov[4].d[0] + rhov[4].d[2];                         // rhov[1][0] = (rhov[4][0] + rhov[4][1]) + (rhov[4][2] + rhov[4][3])
    rhov[1].d[1] = rhov[5].d[0] + rhov[5].d[2];                         // rhov[1][1] = (rhov[5][0] + rhov[5][1]) + (rhov[5][2] + rhov[5][3])
    rhov[1].d[2] = rhov[6].d[0] + rhov[6].d[2];                         // rhov[1][2] = (rhov[6][0] + rhov[6][1]) + (rhov[6][2] + rhov[6][3])

    // yv1 =  alpha * rho + [beta * y_buf[4:6]]
    yv1.v = _mm256_fmadd_pd( alphav.v, rhov[1].v, yv1.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_storeu_pd( y_buf, yv0.v );                               // y_buf[0:3] = yv0
        _mm256_maskstore_pd( y_buf + 4, n_mask, yv1.v );                // y_buf[4:6] = yv1
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
        *( y_buf + 2 * incy ) = yv0.d[2];
        *( y_buf + 3 * incy ) = yv0.d[3];

        *( y_buf + 4 * incy ) = yv1.d[0];
        *( y_buf + 5 * incy ) = yv1.d[1];
        *( y_buf + 6 * incy ) = yv1.d[2];
    }
}

void  bli_dgemv_t_zen_int_mx6_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

     dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // Mask used to load elements 4 + 2 in vector y.
    __m256i n_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[2]);

    // vector variables used load inputs.
    v4df_t yv0, yv1;                                                   // yv0, yv1  -->   Stores 6 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[6];                                                    // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] & rhov[1] --> Result for (a_buf[:, 0:5] * x[:])
    v4df_t a_vec[6];                                                   // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 6 rows of A (columns of A^T).
    double *restrict av[6];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[2,:]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[3,:]
    av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[4,:]
    av[5] = a_buf + 5 * lda;                                           // av[5] = a_buf[5,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();
    rhov[4].v = _mm256_setzero_pd();
    rhov[5].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 and yv1 are assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
        yv1.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf ) );                    // yv0 = beta * y_buf[0:3]
        yv1.v =  _mm256_mul_pd( betav.v, _mm256_maskload_pd( y_buf + 4, n_mask ) );     // yv1 = beta * y_buf[4:5]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy);                                // yv0[3] = y_buf[3]
        yv1.d[0] = *( y_buf + 4 * incy);                                // yv1[0] = y_buf[4]
        yv1.d[1] = *( y_buf + 5 * incy);                                // yv1[1] = y_buf[5]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:3]
        yv1.v =  _mm256_mul_pd( betav.v, yv1.v );                       // yv1 = beta * yv1 = beta * y_buf[4:5]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0:5])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:11, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:11, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:11, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[8:11, 3] * x_buf[8:11]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[8:11, 4] * x_buf[8:11]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[8:11, 5] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 12 );                    // a_vec[3] = a_buf[12:15, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 12 );                    // a_vec[4] = a_buf[12:15, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 12 );                    // a_vec[5] = a_buf[12:15, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 3] * x_buf[12:15]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[12:15, 4] * x_buf[12:15]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[12:15, 5] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        av[5] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:5])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:3, 5] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] + 4 );                     // a_vec[5] = a_buf[4:7, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[4:7, 5] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;
        av[4] += 8;
        av[5] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles (x_buf[0:3] * a_buf[0:3,0:5])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]
        a_vec[5].v =  _mm256_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:3, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[4, 0:7] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[5, 0:7] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        av[2] += 4;
        av[3] += 4;
        av[4] += 4;
        av[5] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:5])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[3].v = _mm256_maskload_pd(av[3], m_mask);                 // a_vec[3] = a_buf[0:m_left, 3]
        a_vec[4].v = _mm256_maskload_pd(av[4], m_mask);                 // a_vec[4] = a_buf[0:m_left, 4]
        a_vec[5].v = _mm256_maskload_pd(av[5], m_mask);                 // a_vec[5] = a_buf[0:m_left, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:3]
        rhov[5].v = _mm256_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 6 vectors (rhov[0:5]),
    // and store the result into the 4 elements of rhov[0] and rhov[1] vectors each.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]
    rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                 // rhov[2][0:1] = rhov[2][0] + rhov[2][1]  &  rhov[2][2:3] = rhov[2][2] + rhov[2][3]
    rhov[3].v = _mm256_hadd_pd( rhov[3].v, rhov[3].v );                 // rhov[3][0:1] = rhov[3][0] + rhov[3][1]  &  rhov[3][2:3] = rhov[3][2] + rhov[3][3]
    rhov[4].v = _mm256_hadd_pd( rhov[4].v, rhov[4].v );                 // rhov[4][0:1] = rhov[4][0] + rhov[4][1]  &  rhov[4][2:3] = rhov[4][2] + rhov[4][3]
    rhov[5].v = _mm256_hadd_pd( rhov[5].v, rhov[5].v );                 // rhov[5][0:1] = rhov[5][0] + rhov[5][1]  &  rhov[5][2:3] = rhov[5][2] + rhov[5][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
    rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])
    rhov[0].d[3] = rhov[3].d[0] + rhov[3].d[2];                         // rhov[0][3] = (rhov[3][0] + rhov[3][1]) + (rhov[3][2] + rhov[3][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:3]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[1].d[0] = rhov[4].d[0] + rhov[4].d[2];                         // rhov[1][0] = (rhov[4][0] + rhov[4][1]) + (rhov[4][2] + rhov[4][3])
    rhov[1].d[1] = rhov[5].d[0] + rhov[5].d[2];                         // rhov[1][1] = (rhov[5][0] + rhov[5][1]) + (rhov[5][2] + rhov[5][3])

    // yv1 =  alpha * rho + [beta * y_buf[4:5]]
    yv1.v = _mm256_fmadd_pd( alphav.v, rhov[1].v, yv1.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_storeu_pd( y_buf, yv0.v );                               // y_buf[0:3] = yv0
        _mm256_maskstore_pd( y_buf + 4, n_mask, yv1.v );                // y_buf[4:5] = yv1
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
        *( y_buf + 2 * incy ) = yv0.d[2];
        *( y_buf + 3 * incy ) = yv0.d[3];
        *( y_buf + 4 * incy ) = yv1.d[0];
        *( y_buf + 5 * incy ) = yv1.d[1];
    }
}

void  bli_dgemv_t_zen_int_mx5_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // vector variables used load inputs.
    v4df_t yv0, yv1;                                                   // yv0, yv1  -->   Stores 5 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[5];                                                    // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] & rhov[1] --> Result for (a_buf[:, 0:4] * x[:])
    v4df_t a_vec[5];                                                   // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 5 rows of A (columns of A^T).
    double *restrict av[5];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[2,:]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[3,:]
    av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[4,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();
    rhov[4].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 and yv1 are assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
        yv1.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf ) );    // yv0 = beta * y_buf[0:3]
        yv1.d[0] = *( y_buf + 4 * incy);                                // yv1[0] = y_buf[4]
        yv1.v =  _mm256_mul_pd( betav.v, yv1.v );                       // yv1 = beta * yv1 = beta * y_buf[4:6]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy);                                // yv0[3] = y_buf[3]

        yv1.d[0] = *( y_buf + 4 * incy);                                // yv1[0] = y_buf[4]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:3]
        yv1.v =  _mm256_mul_pd( betav.v, yv1.v );                       // yv1 = beta * yv1 = beta * y_buf[4:6]
    }


    // Handles (x_buf[0:15] * a_buf[0:15,0:4])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:11, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:11, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[8:11, 3] * x_buf[8:11]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[8:11, 4] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 12 );                    // a_vec[3] = a_buf[12:15, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 12 );                    // a_vec[4] = a_buf[12:15, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 3] * x_buf[12:15]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[12:15, 4] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:4])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:3, 4] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] + 4 );                     // a_vec[4] = a_buf[4:7, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[4:7, 4] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;
        av[4] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles (x_buf[0:3] * a_buf[0:3,0:4])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]
        a_vec[4].v =  _mm256_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:3, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[4, 0:7] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        av[2] += 4;
        av[3] += 4;
        av[4] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:4])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v = _mm256_maskload_pd(av[3], m_mask);                 // a_vec[3] = a_buf[0:m_left, 3]
        a_vec[4].v = _mm256_maskload_pd(av[4], m_mask);                 // a_vec[4] = a_buf[0:m_left, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:3]
        rhov[4].v = _mm256_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 5 vectors (rhov[0:4]),
    // and store the result into the 4 elements of rhov[0] and rhov[1] vectors each.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]
    rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                 // rhov[2][0:1] = rhov[2][0] + rhov[2][1]  &  rhov[2][2:3] = rhov[2][2] + rhov[2][3]
    rhov[3].v = _mm256_hadd_pd( rhov[3].v, rhov[3].v );                 // rhov[3][0:1] = rhov[3][0] + rhov[3][1]  &  rhov[3][2:3] = rhov[3][2] + rhov[3][3]
    rhov[4].v = _mm256_hadd_pd( rhov[4].v, rhov[4].v );                 // rhov[4][0:1] = rhov[4][0] + rhov[4][1]  &  rhov[4][2:3] = rhov[4][2] + rhov[4][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
    rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])
    rhov[0].d[3] = rhov[3].d[0] + rhov[3].d[2];                         // rhov[0][3] = (rhov[3][0] + rhov[3][1]) + (rhov[3][2] + rhov[3][3])
    rhov[1].d[0] = rhov[4].d[0] + rhov[4].d[2];                         // rhov[1][0] = (rhov[4][0] + rhov[4][1]) + (rhov[4][2] + rhov[4][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:3]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // yv1 =  alpha * rho + [beta * y_buf[4]]
    yv1.v = _mm256_fmadd_pd( alphav.v, rhov[1].v, yv1.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_storeu_pd( y_buf, yv0.v );                               // y_buf[0:3] = yv0
        *( y_buf + 4 * incy ) = yv1.d[0];                               // y_buf[4] = yv1
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
        *( y_buf + 2 * incy ) = yv0.d[2];
        *( y_buf + 3 * incy ) = yv0.d[3];
        *( y_buf + 4 * incy ) = yv1.d[0];
    }
}

void  bli_dgemv_t_zen_int_mx4_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

     dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // vector variables used load inputs.
    v4df_t yv0;                                                        // yv0  -->   Stores 4 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[4];                                                    // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] --> Result for (a_buf[:, 0:3] * x[:])
    v4df_t a_vec[4];                                                   // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 4 rows of A (columns of A^T).
    double *restrict av[4];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[2,:]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[3,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 is assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_loadu_pd( y_buf ) );    // yv0 = beta * y_buf[0:3]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy);                                // yv0[3] = y_buf[3]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:3]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0:3])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:11, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[8:11, 3] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 12 );                    // a_vec[3] = a_buf[12:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 3] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:3])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] + 4 );                     // a_vec[3] = a_buf[4:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[4:7, 3] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles (x_buf[0:3] * a_buf[0:3,0:3])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]
        a_vec[3].v =  _mm256_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:3, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:3, 3] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        av[2] += 4;
        av[3] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:3])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v = _mm256_maskload_pd(av[3], m_mask);                 // a_vec[3] = a_buf[0:m_left, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 4 vectors (rhov[0:3]),
    // and store the result into the 4 elements of rhov[0] vector.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]
    rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                 // rhov[2][0:1] = rhov[2][0] + rhov[2][1]  &  rhov[2][2:3] = rhov[2][2] + rhov[2][3]
    rhov[3].v = _mm256_hadd_pd( rhov[3].v, rhov[3].v );                 // rhov[3][0:1] = rhov[3][0] + rhov[3][1]  &  rhov[3][2:3] = rhov[3][2] + rhov[3][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
    rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])
    rhov[0].d[3] = rhov[3].d[0] + rhov[3].d[2];                         // rhov[0][3] = (rhov[3][0] + rhov[3][1]) + (rhov[3][2] + rhov[3][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:3]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_storeu_pd( y_buf, yv0.v );                               // y_buf[0:3] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
        *( y_buf + 2 * incy ) = yv0.d[2];
        *( y_buf + 3 * incy ) = yv0.d[3];
    }
}

void  bli_dgemv_t_zen_int_mx3_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // Mask used to load elements 3 in vector y.
    __m256i n_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[3]);

    // vector variables used load inputs.
    v4df_t yv0;                                                        // yv0, yv1  -->   Stores 3 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[3];                                                    // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])     rhov[0] --> Result for (a_buf[:, 0:2] * x[:])
    v4df_t a_vec[3];                                                   // a_vec[i] --> a_buf[0:3, i]

    // Set up pointers for 3 rows of A (columns of A^T).
    double *restrict av[3];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[2,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 is assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_maskload_pd( y_buf, n_mask ) );    // yv0 = beta * y_buf[0:2]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy);                                // yv0[2] = y_buf[2]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:2]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0:2])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:11, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 2] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 12 );                    // a_vec[2] = a_buf[12:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[12:15, 2] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:2])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] + 4 );                     // a_vec[2] = a_buf[4:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[4:7, 2] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles (x_buf[0:3] * a_buf[0:3,0:2])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]
        a_vec[2].v =  _mm256_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:3, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:3, 2] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        av[2] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:2])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v = _mm256_maskload_pd(av[2], m_mask);                 // a_vec[2] = a_buf[0:m_left, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 3 vectors (rhov[0:2]),
    // and store the result into the 4 elements of rhov[0] vector.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]
    rhov[2].v = _mm256_hadd_pd( rhov[2].v, rhov[2].v );                 // rhov[2][0:1] = rhov[2][0] + rhov[2][1]  &  rhov[2][2:3] = rhov[2][2] + rhov[2][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])
    rhov[0].d[2] = rhov[2].d[0] + rhov[2].d[2];                         // rhov[0][2] = (rhov[2][0] + rhov[2][1]) + (rhov[2][2] + rhov[2][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:2]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_maskstore_pd( y_buf, n_mask, yv0.v );                               // y_buf[0:2] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
        *( y_buf + 2 * incy ) = yv0.d[2];
    }
}

void  bli_dgemv_t_zen_int_mx2_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                           //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // Mask used to load elements 2 in vector y.
    __m256i n_mask = _mm256_loadu_si256( (__m256i *)mask_ptr[2]);

    // vector variables used load inputs.
    v4df_t yv0;                                                        // yv0  -->   Stores 2 elements from vector y
    v4df_t xv0, xv1, xv2, xv3;                                         // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[4];                                                    // rhov[0] & rhov[2] --> Accumulator for (a_buf[:, 0:1] * x_buf[:])    rhov[1] & rhov[3] --> Accumulator for (a_buf[:, 1] * x_buf[:])
    v4df_t a_vec[4];                                                   // a_vec[0] --> a_buf[0:3, 0]    a_vec[1] --> a_buf[0:3, 1]    a_vec[2] --> a_buf[4:7, 0]    a_vec[0] --> a_buf[4:7, 1]

    // Set up pointers for 2 rows of A (columns of A^T).
    double *restrict av[2];
    v4df_t alphav;
    v4df_t betav;

    alphav.v = _mm256_set1_pd( *alpha );
    betav.v = _mm256_set1_pd( *beta );

    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[0,:]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[1,:]

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv0 is assigned zero if beta is 0
        yv0.v = _mm256_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm5256_mul_pd' to get beta * y
        yv0.v =  _mm256_mul_pd( betav.v, _mm256_maskload_pd( y_buf, n_mask ) );    // yv0 = beta * y_buf[0:1]
    }
    else
    {
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]

        yv0.v = _mm256_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:1]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0:1])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[3].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]

        // Load the input values from vector X.
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:11, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:11, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[8:11, 1] * x_buf[8:11]

        // Load the input values from vector X.
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm256_loadu_pd( av[0] + 12 );                    // a_vec[0] = a_buf[12:15, 0]
        a_vec[3].v =  _mm256_loadu_pd( av[1] + 12 );                    // a_vec[1] = a_buf[12:15, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[0] += a_buf[12:15, 0] * x_buf[12:15]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[1] += a_buf[12:15, 1] * x_buf[12:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        x_buf += 16;
    }

    // Handles (x_buf[0:7] * a_buf[0:7,0:1])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]

        // Load the input values from vector X.
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm256_loadu_pd( av[0] + 4 );                     // a_vec[0] = a_buf[4:7, 0]
        a_vec[3].v =  _mm256_loadu_pd( av[1] + 4 );                     // a_vec[1] = a_buf[4:7, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[0] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[1] += a_buf[4:7, 1] * x_buf[4:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        x_buf += 8;
        i     += 8;
    }

    rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[2].v);                    // rhov[0] = rhov[0] + rhov[2]
    rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[3].v);                    // rhov[1] = rhov[1] + rhov[3]

    // Handles (x_buf[0:3] * a_buf[0:3,0:1])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:3, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:3, 1] * x_buf[0:3]

        // Incrementing pointers
        av[0] += 4;
        av[1] += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0:1])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                      // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(av[0], m_mask);                 // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v = _mm256_maskload_pd(av[1], m_mask);                 // a_vec[1] = a_buf[0:m_left, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in 2 vectors (rhov[0:1]),
    // and store the result into the 4 elements of rhov[0] vector.
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );                 // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[1].v = _mm256_hadd_pd( rhov[1].v, rhov[1].v );                 // rhov[1][0:1] = rhov[1][0] + rhov[1][1]  &  rhov[1][2:3] = rhov[1][2] + rhov[1][3]

    // Sum the results of the horizontal adds to get the final sums for each vector
    rhov[0].d[0] = rhov[0].d[0] + rhov[0].d[2];                         // rhov[0][0] = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rhov[0].d[1] = rhov[1].d[0] + rhov[1].d[2];                         // rhov[0][1] = (rhov[1][0] + rhov[1][1]) + (rhov[1][2] + rhov[1][3])

    // yv0 =  alpha * rho + [beta * y_buf[0:1]]
    yv0.v = _mm256_fmadd_pd( alphav.v, rhov[0].v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm256_storeu_pd'
        _mm256_maskstore_pd( y_buf, n_mask, yv0.v );                               // y_buf[0:1] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy ) = yv0.d[0];
        *( y_buf + 1 * incy ) = yv0.d[1];
    }
}

void  bli_dgemv_t_zen_int_mx1_avx2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t cs, inc_t rs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict a_buf = a;
    double* restrict y_buf = y;
    double* restrict x_buf = x;

    // i denotes the number of columns completed
    dim_t i = 0;

    int64_t mask_4[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_4  -->  [  x  x  x  x  ]
            mask_4[0] = -1;
            mask_4[1] = -1;
            mask_4[2] = -1;
            mask_4[3] = -1;

    int64_t mask_3[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_3  -->  [  x  x  x  _  ]
            mask_3[0] = -1;
            mask_3[1] = -1;
            mask_3[2] = -1;
            mask_3[3] = 0;

    int64_t mask_2[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_2  -->  [  x  x  _  _  ]
            mask_2[0] = -1;
            mask_2[1] = -1;
            mask_2[2] = 0;
            mask_2[3] = 0;

    int64_t mask_1[4] = {0};                                                //  [  x  x  x  x  ]  -->  mask_1  -->  [  x  _  _  _  ]
            mask_1[0] = -1;
            mask_1[1] = 0;
            mask_1[2] = 0;
            mask_1[3] = 0;

    int64_t *mask_ptr[] = {mask_4, mask_1, mask_2, mask_3, mask_4};

    dim_t m_left = m % 4;

    // Mask used to load x, when fringe cases are considered.
    __m256i m_mask  = _mm256_loadu_si256( (__m256i *)mask_ptr[m_left]);

    // vector variables used load inputs.
    v4df_t xv0, xv1, xv2, xv3;                                              // xv0 --> x_buf[0:3], xv1 --> x_buf[4:7], xv2 --> x_buf[8:11], xv3 --> x_buf[12:15]
    v4df_t rhov[4];                                                         // rhov[0:3]--> Accumulator for (a_buf[:, 0] * x_buf[:])
    v4df_t a_vec[4];                                                        // a_vec[0] --> a_buf[0:3, 0]    a_vec[1] --> a_buf[4:7, 0]    a_vec[2] --> a_buf[8:11, 0]    a_vec[3] --> a_buf[12:15, 0]

    // declaring double precision varibles to store the result
    double yv, rh;                                                           // yv0 --> y_buf[0]   rh  --> Result for (a_buf[:, 0] * x[:])

    rhov[0].v = _mm256_setzero_pd();
    rhov[1].v = _mm256_setzero_pd();
    rhov[2].v = _mm256_setzero_pd();
    rhov[3].v = _mm256_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {

        // yv is assigned zero if beta is 0
        yv = 0;
    }
    else
    {
        // The inputs on vector y are manually moved to registor yv0 and multipled by beta
        yv = (*beta) * (*y_buf);                                        // yv0 = beta * y_buf[0]
    }

    // Handles (x_buf[0:15] * a_buf[0:15,0])
    for ( i = 0; (i + 15) < m; i += 16 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]
        xv2.v =  _mm256_loadu_pd( x_buf + 8 );                          // xv2 = x_buf[8:11]
        xv3.v =  _mm256_loadu_pd( x_buf + 12 );                         // xv3 = x_buf[12:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( a_buf + 4 );                     // a_vec[1] = a_buf[4:7, 0]
        a_vec[2].v =  _mm256_loadu_pd( a_buf + 8 );                     // a_vec[2] = a_buf[8:11, 0]
        a_vec[3].v =  _mm256_loadu_pd( a_buf + 12 );                    // a_vec[3] = a_buf[12:15, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 0] * x_buf[4:7]
        rhov[2].v = _mm256_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[8:11, 0] * x_buf[8:11]
        rhov[3].v = _mm256_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[12:15, 0] * x_buf[12:15]

        // Incrementing pointers
        a_buf += 16;
        x_buf += 16;
    }

    rhov[0].v = _mm256_add_pd(rhov[0].v, rhov[2].v);                    // rhov[0] = rhov[0] + rhov[2]
    rhov[1].v = _mm256_add_pd(rhov[1].v, rhov[3].v);                    // rhov[1] = rhov[1] + rhov[3]

    // Handles (x_buf[0:7] * a_buf[0:7,0])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]
        xv1.v =  _mm256_loadu_pd( x_buf + 4 );                          // xv1 = x_buf[4:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:3, 0]
        a_vec[1].v =  _mm256_loadu_pd( a_buf + 4 );                     // a_vec[1] = a_buf[4:7, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]
        rhov[1].v = _mm256_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[4:7, 0] * x_buf[4:7]

        // Incrementing pointers
        a_buf += 8;
        x_buf += 8;
        i     += 8;
    }

    rhov[0].v = _mm256_add_pd( rhov[0].v, rhov[1].v );                  // rhov[0] = rhov[0] + rhov[1]

    // Handles (x_buf[0:3] * a_buf[0:3,0])
    if ( (i + 3) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm256_loadu_pd( x_buf );                              // xv0 = x_buf[0:3]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm256_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:3, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:3, 0] * x_buf[0:3]

        // Incrementing pointers
        a_buf += 4;
        x_buf += 4;
        i     += 4;
    }

    // Handles fringe cases -> (x_buf[0:m_left] * a_buf[0:m_left,0])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v = _mm256_maskload_pd(x_buf, m_mask);                          // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v = _mm256_maskload_pd(a_buf, m_mask);                     // a_vec[0] = a_buf[0:m_left, 0]

        // Perform: rho?v += a?v * x0v;
        rhov[0].v = _mm256_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );        // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:3]
    }
    // This section of code is used to find the sum of values in vector rhov[0],
    // and store the result into the 4 elements of rhov[0] vector.
    // rhov[0][0:1] = rhov[0][0] + rhov[0][1]  &  rhov[0][2:3] = rhov[0][2] + rhov[0][3]
    rhov[0].v = _mm256_hadd_pd( rhov[0].v, rhov[0].v );

    // rh = (rhov[0][0] + rhov[0][1]) + (rhov[0][2] + rhov[0][3])
    rh = rhov[0].d[0] + rhov[0].d[2];

    // Perform y = alpha*x + beta*y;
    yv += ((*alpha) * rh);

    // Store the result back into vector y
    *( y_buf ) = yv;

}
