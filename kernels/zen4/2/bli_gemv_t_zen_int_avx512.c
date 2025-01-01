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

/* Union data structure to access AVX-512 registers
*  One 512-bit AVX register holds 8 DP elements. */
typedef union
{
  __m512d v;
  double  d[8] __attribute__((aligned(64)));
} v8df_t;

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
    bli_dgemv_t_zen_int_mx1_avx512,   // n = 1
    bli_dgemv_t_zen_int_mx2_avx512,   // n = 2
    bli_dgemv_t_zen_int_mx3_avx512,   // n = 3
    bli_dgemv_t_zen_int_mx4_avx512,   // n = 4
    bli_dgemv_t_zen_int_mx5_avx512,   // n = 5
    bli_dgemv_t_zen_int_mx6_avx512,   // n = 6
    bli_dgemv_t_zen_int_mx7_avx512    // n = 7
};

/**
 * bli_dgemv_t_zen_int_avx512(...) handles cases where op(A) = TRANSPOSE && column-storage
 * or op(A) = NON-TRANSPOSE && row-storage. We will compute 8 columns of A at a time until
 * less than 8 columns remain. Then we will then call others kernels to handle fringe cases.
 *
 * Here we will use dot product to multiply each column of matrix A with vector x in
 * groups of 8 columns, resulting product will be stored in a temporary vector before
 * being added to vector y.
 *
 *  In case of non-transpose row storage, the values of m, n, inca and lda will be interchanged
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

void bli_dgemv_t_zen_int_avx512
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

    // n_iter is the number of iterations of the loop that need to be executed
    // n_left is the number of columns remaining after the loop is completed
    // m_left is the number of rows remaining in the fringe case inside the for loop
    dim_t n_iter = n / 8;
    dim_t n_left = n % 8;
    dim_t m_left = m % 8;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:7]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[8];                                                     // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t rho;                                                         // rho --> Result for (a_buf[:, 0:7] * x[:])
    v8df_t a_vec[8];                                                    // a_vec[i] --> a_buf[0:7, i]

    // Set up pointers for 8 rows of A (columns of A^T).
    double *restrict av[8];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );

    // The following loop processes the matrix A in chunks of 8 columns at a time.
    // For each chunk, the result of the matrix-vector multiplication is stored in the corresponding elements of the vector y.
    for ( i = 0; i < n_iter; ++i)
    {

        //The variable 'j' is used to denote the number of rows that have been completed.
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
        rhov[0].v = _mm512_setzero_pd();
        rhov[1].v = _mm512_setzero_pd();
        rhov[2].v = _mm512_setzero_pd();
        rhov[3].v = _mm512_setzero_pd();

        rhov[4].v = _mm512_setzero_pd();
        rhov[5].v = _mm512_setzero_pd();
        rhov[6].v = _mm512_setzero_pd();
        rhov[7].v = _mm512_setzero_pd();

        // Loading data from vector y
        if (bli_deq0( *beta ))
        {
            // yv is assigned zero if beta is 0
            yv0.v = _mm512_setzero_pd();
        }
        else if ( incy == 1)
        {
            // In case of unit stride y, the inputs on vector are moved to register yv using vector load
            // Followed by '_mm512_mul_pd' to get beta * y
            yv0.v = _mm512_mul_pd( betav.v, _mm512_loadu_pd( y_buf ) );     // yv0 = beta * y_buf[0:7]
        }
        else
        {
            // In case of non-unit stride y,
            // The inputs on vector y are manually moved to registor yv0
            yv0.d[0] = *( y_buf + 0 * incy );                               // yv0[0] = y_buf[0]
            yv0.d[1] = *( y_buf + 1 * incy );                               // yv0[1] = y_buf[1]
            yv0.d[2] = *( y_buf + 2 * incy );                               // yv0[2] = y_buf[2]
            yv0.d[3] = *( y_buf + 3 * incy );                               // yv0[3] = y_buf[3]

            yv0.d[4] = *( y_buf + 4 * incy );                               // yv0[4] = y_buf[4]
            yv0.d[5] = *( y_buf + 5 * incy );                               // yv0[5] = y_buf[5]
            yv0.d[6] = *( y_buf + 6 * incy );                               // yv0[6] = y_buf[6]
            yv0.d[7] = *( y_buf + 7 * incy );                               // yv0[7] = y_buf[7]

            yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:7]
        }

        // Handles (a_buf[0:31, 0:7] * x_buf[0:31])
        for ( j = 0; (j + 31) < m; j += 32 )
        {
            // Load the input values from vector X.
            xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:7, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:7, 7] * x_buf[0:7]

            // Load the input values from vector X.
            xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:15, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] + 8 );                     // a_vec[7] = a_buf[8:15, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[8:15, 6] * x_buf[8:15]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv1.v, rhov[7].v );    // rhov[7] += a_buf[8:15, 7] * x_buf[8:15]

            // Load the input values from vector X.
            xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                    // a_vec[2] = a_buf[16:23, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] + 16 );                    // a_vec[3] = a_buf[16:23, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[16:23, 3] * x_buf[16:23]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] + 16 );                    // a_vec[4] = a_buf[16:23, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] + 16 );                    // a_vec[5] = a_buf[16:23, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] + 16 );                    // a_vec[6] = a_buf[16:23, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] + 16 );                    // a_vec[7] = a_buf[16:23, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[16:23, 4] * x_buf[16:23]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[16:23, 5] * x_buf[16:23]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv2.v, rhov[6].v );    // rhov[6] += a_buf[16:23, 6] * x_buf[16:23]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv2.v, rhov[7].v );    // rhov[7] += a_buf[16:23, 7] * x_buf[16:23]

            // Load the input values from vector X.
            xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                    // a_vec[2] = a_buf[24:31, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] + 24 );                    // a_vec[3] = a_buf[24:31, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 3] * x_buf[24:31]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] + 24 );                    // a_vec[4] = a_buf[24:31, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] + 24 );                    // a_vec[5] = a_buf[24:31, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] + 24 );                    // a_vec[6] = a_buf[24:31, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] + 24 );                    // a_vec[7] = a_buf[24:31, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[24:31, 4] * x_buf[24:31]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[24:31, 5] * x_buf[24:31]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv3.v, rhov[6].v );    // rhov[6] += a_buf[24:31, 6] * x_buf[24:31]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv3.v, rhov[7].v );    // rhov[7] += a_buf[24:31, 7] * x_buf[24:31]

            // Incrementing pointers by 32 (4 iterations * 8 elements per register)
            av[0] += 32;
            av[1] += 32;
            av[2] += 32;
            av[3] += 32;
            av[4] += 32;
            av[5] += 32;
            av[6] += 32;
            av[7] += 32;
            x_buf += 32;
        }

        // Handles (a_buf[0:15, 0:7] * x_buf[0:15])
        if ( (j + 15) < m )
        {
            // Load the input values from vector X.
            xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:7, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:7, 7] * x_buf[0:7]

            // Load the input values from vector X.
            xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:15, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] + 8 );                     // a_vec[7] = a_buf[8:15, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[8:15, 6] * x_buf[8:15]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv1.v, rhov[7].v );    // rhov[7] += a_buf[8:15, 7] * x_buf[8:15]

            // Incrementing pointers by 16 (2 iterations * 8 elements per register)
            av[0] += 16;
            av[1] += 16;
            av[2] += 16;
            av[3] += 16;
            av[4] += 16;
            av[5] += 16;
            av[6] += 16;
            av[7] += 16;
            x_buf += 16;
            j     += 16;
        }

        // Handles (a_buf[0:7, 0:7] * x_buf[0:7])
        if ( (j + 7) < m )
        {
            // Load the input values from vector X.
            xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
            a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
            a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
            a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
            a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
            a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]
            a_vec[7].v =  _mm512_loadu_pd( av[7] );                         // a_vec[7] = a_buf[0:7, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:7, 7] * x_buf[0:7]

            // Incrementing pointers by 8 (1 iteration * 8 elements per register)
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

        // Handles fringe cases -> (a_buf[0:m_left, 0:7] * x_buf[0:m_left])
        if( m_left )
        {
            // Load the input values from vector X.
            xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

            // Load the input values from Matrix A
            a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
            a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
            a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]
            a_vec[3].v =  _mm512_maskz_loadu_pd( m_mask, av[3] );           // a_vec[3] = a_buf[0:m_left, 3]

            // perform: rho?v += a?v * x0v;
            rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
            rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
            rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]
            rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:m_left]

            // Load the input values from Matrix A
            a_vec[4].v =  _mm512_maskz_loadu_pd( m_mask, av[4] );           // a_vec[4] = a_buf[0:m_left, 4]
            a_vec[5].v =  _mm512_maskz_loadu_pd( m_mask, av[5] );           // a_vec[5] = a_buf[0:m_left, 5]
            a_vec[6].v =  _mm512_maskz_loadu_pd( m_mask, av[6] );           // a_vec[6] = a_buf[0:m_left, 6]
            a_vec[7].v =  _mm512_maskz_loadu_pd( m_mask, av[7] );           // a_vec[7] = a_buf[0:m_left, 7]

            // perform: rho?v += a?v * x0v;
            rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:m_left]
            rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:m_left]
            rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:m_left, 6] * x_buf[0:m_left]
            rhov[7].v = _mm512_fmadd_pd( a_vec[7].v, xv0.v, rhov[7].v );    // rhov[7] += a_buf[0:m_left, 7] * x_buf[0:m_left]
        }

        // This section of code is used to find the sum of  values in 8 vectors (rhov[0:7]),
        // and store the reesult into the 8 elements of rho vector.
        rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
        rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
        rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]
        rho.d[3] = _mm512_reduce_add_pd( rhov[3].v );                       // rho[3] = a_buf[3, 0:m] * x_buf[0:m]

        rho.d[4] = _mm512_reduce_add_pd( rhov[4].v );                       // rho[4] = a_buf[4, 0:m] * x_buf[0:m]
        rho.d[5] = _mm512_reduce_add_pd( rhov[5].v );                       // rho[5] = a_buf[5, 0:m] * x_buf[0:m]
        rho.d[6] = _mm512_reduce_add_pd( rhov[6].v );                       // rho[6] = a_buf[6, 0:m] * x_buf[0:m]
        rho.d[7] = _mm512_reduce_add_pd( rhov[7].v );                       // rho[7] = a_buf[7, 0:m] * x_buf[0:m]

        // yv0 =  alpha * rho + [beta * y_buf[0:7]]
        yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

        // Store the result back into vector y
        if ( incy == 1)
        {
            // In case of unit stride y,
            // The result is moved back onto vector y using '_mm512_storeu_pd'
            _mm512_storeu_pd( y_buf , yv0.v );                              // y_buf[0:7] = yv0
        }
        else
        {
            // In case of non-unit stride y,
            // The result is manually moved back onto vector y
            *( y_buf + 0 * incy) = yv0.d[0];
            *( y_buf + 1 * incy) = yv0.d[1];
            *( y_buf + 2 * incy) = yv0.d[2];
            *( y_buf + 3 * incy) = yv0.d[3];

            *( y_buf + 4 * incy) = yv0.d[4];
            *( y_buf + 5 * incy) = yv0.d[5];
            *( y_buf + 6 * incy) = yv0.d[6];
            *( y_buf + 7 * incy) = yv0.d[7];
        }

        // The pointers are moved to the corresponding position for next calculation.
        x_buf = x;
        y_buf += 8 * incy;
        a_buf += 8 * lda;
    }

    // Handle the remaining columns (fringe cases) if n_left is not zero.
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

void  bli_dgemv_t_zen_int_mx7_avx512
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

    // Number of fringe cases
    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    // Mask used to load elements 7 in vector y.
    __mmask8 n_mask  = (1 << (7)) - 1;

    // Vector variables used to load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:6]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[7];                                                     // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t a_vec[7];                                                    // a_vec[i] --> a_buf[0:7, i]
    v8df_t rho;                                                         // rho --> Result for (a_buf[:, 0:6] * x[:])

    // Set up pointers for 7 rows of A (columns of A^T).
    double *restrict av[7];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v  = _mm512_set1_pd( *beta );
    rho.v    = _mm512_setzero_pd();

    // Creating an array of pointers for 7 columns of matrix A
    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[:, 1]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[:, 2]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[:, 3]
    av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[:, 4]
    av[5] = a_buf + 5 * lda;                                           // av[5] = a_buf[:, 5]
    av[6] = a_buf + 6 * lda;                                           // av[6] = a_buf[:, 6]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();
    rhov[4].v = _mm512_setzero_pd();
    rhov[5].v = _mm512_setzero_pd();
    rhov[6].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:6]
    }
    else
    {
        yv0.v = _mm512_setzero_pd();
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy );                               // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy );                               // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy );                               // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy );                               // yv0[3] = y_buf[3]
        yv0.d[4] = *( y_buf + 4 * incy );                               // yv0[4] = y_buf[4]
        yv0.d[5] = *( y_buf + 5 * incy );                               // yv0[5] = y_buf[5]
        yv0.d[6] = *( y_buf + 6 * incy );                               // yv0[6] = y_buf[6]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:6]
    }

    // Handles (a_buf[0:31, 0:6] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:15, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[8:15, 6] * x_buf[8:15]

        // Load the input values from vector X.
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                    // a_vec[2] = a_buf[16:23, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 16 );                    // a_vec[3] = a_buf[16:23, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[16:23, 3] * x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 16 );                    // a_vec[4] = a_buf[16:23, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 16 );                    // a_vec[5] = a_buf[16:23, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] + 16 );                    // a_vec[6] = a_buf[16:23, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[16:23, 4] * x_buf[16:23]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[16:23, 5] * x_buf[16:23]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv2.v, rhov[6].v );    // rhov[6] += a_buf[16:23, 6] * x_buf[16:23]

        // Load the input values from vector X.
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                    // a_vec[2] = a_buf[24:31, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 24 );                    // a_vec[3] = a_buf[24:31, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 3] * x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 24 );                    // a_vec[4] = a_buf[24:31, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 24 );                    // a_vec[5] = a_buf[24:31, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] + 24 );                    // a_vec[6] = a_buf[24:31, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[24:31, 4] * x_buf[24:31]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[24:31, 5] * x_buf[24:31]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv3.v, rhov[6].v );    // rhov[6] += a_buf[24:31, 6] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        av[2] += 32;
        av[3] += 32;
        av[4] += 32;
        av[5] += 32;
        av[6] += 32;
        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:6] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] + 8 );                     // a_vec[6] = a_buf[8:15, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv1.v, rhov[6].v );    // rhov[6] += a_buf[8:15, 6] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        av[5] += 16;
        av[6] += 16;
        x_buf += 16;
        i     += 16;
    }

    // Handles ( a_buf[0:7, 0:6] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]
        a_vec[6].v =  _mm512_loadu_pd( av[6] );                         // a_vec[6] = a_buf[0:7, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:7, 6] * x_buf[0:7]

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

    // Handles fringe cases -> (a_buf[0:m_left, 0:6] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v =  _mm512_maskz_loadu_pd( m_mask, av[3] );           // a_vec[3] = a_buf[0:m_left, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[4].v =  _mm512_maskz_loadu_pd( m_mask, av[4] );           // a_vec[4] = a_buf[0:m_left, 4]
        a_vec[5].v =  _mm512_maskz_loadu_pd( m_mask, av[5] );           // a_vec[5] = a_buf[0:m_left, 5]
        a_vec[6].v =  _mm512_maskz_loadu_pd( m_mask, av[6] );           // a_vec[6] = a_buf[0:m_left, 6]

        // perform: rho?v += a?v * x0v;
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:m_left]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:m_left]
        rhov[6].v = _mm512_fmadd_pd( a_vec[6].v, xv0.v, rhov[6].v );    // rhov[6] += a_buf[0:m_left, 6] * x_buf[0:m_left]
    }

    // This section of code is used to find the sum of  values in 8 vectors (rhov[0:7]),
    // and store the reesult into the 8 elements of rho vector.
    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
    rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]
    rho.d[3] = _mm512_reduce_add_pd( rhov[3].v );                       // rho[3] = a_buf[3, 0:m] * x_buf[0:m]

    rho.d[4] = _mm512_reduce_add_pd( rhov[4].v );                       // rho[4] = a_buf[4, 0:m] * x_buf[0:m]
    rho.d[5] = _mm512_reduce_add_pd( rhov[5].v );                       // rho[5] = a_buf[5, 0:m] * x_buf[0:m]
    rho.d[6] = _mm512_reduce_add_pd( rhov[6].v );                       // rho[6] = a_buf[6, 0:m] * x_buf[0:m]

    // yv0 =  alpha * rho + [beta * y_buf[0:6]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm512_storeu_pd'
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );                 // y_buf[0:6] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
        *( y_buf + 2 * incy) = yv0.d[2];
        *( y_buf + 3 * incy) = yv0.d[3];

        *( y_buf + 4 * incy) = yv0.d[4];
        *( y_buf + 5 * incy) = yv0.d[5];
        *( y_buf + 6 * incy) = yv0.d[6];
    }

}

void  bli_dgemv_t_zen_int_mx6_avx512
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

    // Number of fringe cases
    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    // Mask used to load elements 6 in vector y.
    __mmask8 n_mask  = (1 << (6)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:5]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[6];                                                     // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t a_vec[6];                                                    // a_vec[i] --> a_buf[0:7, i]
    v8df_t rho;                                                         // rho  --> Result for (a_buf[:, 0:5] * x[:])

    // Set up pointers for 6 rows of A (columns of A^T).
    double *restrict av[6];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );
    rho.v = _mm512_setzero_pd();

    // Creating an array of pointers for 6 columns of matrix A
    av[0] = a_buf + 0 * lda;           // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;           // av[1] = a_buf[:, 1]
    av[2] = a_buf + 2 * lda;           // av[2] = a_buf[:, 2]
    av[3] = a_buf + 3 * lda;           // av[3] = a_buf[:, 3]
    av[4] = a_buf + 4 * lda;           // av[4] = a_buf[:, 4]
    av[5] = a_buf + 5 * lda;           // av[5] = a_buf[:, 5]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();
    rhov[4].v = _mm512_setzero_pd();
    rhov[5].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:5]
    }
    else
    {
        yv0.v = _mm512_setzero_pd();
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy );                               // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy );                               // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy );                               // yv0[2] = y_buf[2]

        yv0.d[3] = *( y_buf + 3 * incy );                               // yv0[3] = y_buf[3]
        yv0.d[4] = *( y_buf + 4 * incy );                               // yv0[4] = y_buf[4]
        yv0.d[5] = *( y_buf + 5 * incy );                               // yv0[5] = y_buf[5]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:6]
    }

    // Handles (a_buf[0:31, 0:5] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                              // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]

        // Load the input values from vector X.
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                    // a_vec[2] = a_buf[16:23, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 16 );                    // a_vec[3] = a_buf[16:23, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 16 );                    // a_vec[4] = a_buf[16:23, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 16 );                    // a_vec[5] = a_buf[16:23, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[16:23, 3] * x_buf[16:23]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[16:23, 4] * x_buf[16:23]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv2.v, rhov[5].v );    // rhov[5] += a_buf[16:23, 5] * x_buf[16:23]

        // Load the input values from vector X.
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                    // a_vec[2] = a_buf[24:31, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 24 );                    // a_vec[3] = a_buf[24:31, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 24 );                    // a_vec[4] = a_buf[24:31, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 24 );                    // a_vec[5] = a_buf[24:31, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 3] * x_buf[24:31]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[24:31, 4] * x_buf[24:31]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv3.v, rhov[5].v );    // rhov[5] += a_buf[24:31, 5] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        av[2] += 32;
        av[3] += 32;
        av[4] += 32;
        av[5] += 32;
        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:5] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] + 8 );                     // a_vec[5] = a_buf[8:15, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv1.v, rhov[5].v );    // rhov[5] += a_buf[8:15, 5] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        av[5] += 16;
        x_buf += 16;
        i     += 16;
    }

    // Handles (a_buf[0:7, 0:5] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]
        a_vec[5].v =  _mm512_loadu_pd( av[5] );                         // a_vec[5] = a_buf[0:7, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:7, 5] * x_buf[0:7]

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

    // Handles fringe cases -> (a_buf[0:m_left, 0:5] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[3].v =  _mm512_maskz_loadu_pd( m_mask, av[3] );           // a_vec[3] = a_buf[0:m_left, 3]
        a_vec[4].v =  _mm512_maskz_loadu_pd( m_mask, av[4] );           // a_vec[4] = a_buf[0:m_left, 4]
        a_vec[5].v =  _mm512_maskz_loadu_pd( m_mask, av[5] );           // a_vec[5] = a_buf[0:m_left, 5]

        // perform: rho?v += a?v * x0v;
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:m_left]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:m_left]
        rhov[5].v = _mm512_fmadd_pd( a_vec[5].v, xv0.v, rhov[5].v );    // rhov[5] += a_buf[0:m_left, 5] * x_buf[0:m_left]
    }

    // This section of code is used to find the sum of  values in 6 vectors (rhov[0:5]),
    // and store the reesult into the 6 elements of rho vector.
    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
    rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]
    rho.d[3] = _mm512_reduce_add_pd( rhov[3].v );                       // rho[3] = a_buf[3, 0:m] * x_buf[0:m]
    rho.d[4] = _mm512_reduce_add_pd( rhov[4].v );                       // rho[4] = a_buf[4, 0:m] * x_buf[0:m]
    rho.d[5] = _mm512_reduce_add_pd( rhov[5].v );                       // rho[5] = a_buf[5, 0:m] * x_buf[0:m]

    // yv0 =  alpha * rho + [beta * y_buf[0:5]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm512_storeu_pd'
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );                 // y_buf[0:5] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
        *( y_buf + 2 * incy) = yv0.d[2];
        *( y_buf + 3 * incy) = yv0.d[3];
        *( y_buf + 4 * incy) = yv0.d[4];
        *( y_buf + 5 * incy) = yv0.d[5];
    }
}

void bli_dgemv_t_zen_int_mx5_avx512
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

    // Number of fringe cases
    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    // Mask used to load elements 5 in vector y.
    __mmask8 n_mask  = (1 << (5)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:4]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[5];                                                     // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t a_vec[5];                                                    // a_vec[i] --> a_buf[0:7, i]
    v8df_t rho;                                                         // rho  --> Result for (a_buf[:, 0:4] * x[:])

    // Set up pointers for 5 rows of A (columns of A^T).
    double *restrict av[5];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );
    rho.v = _mm512_setzero_pd();

    // Creating an array of pointers for 5 columns of matrix A
    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[:, 1]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[:, 2]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[:, 3]
    av[4] = a_buf + 4 * lda;                                           // av[4] = a_buf[:, 4]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();
    rhov[4].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:4]
    }
    else
    {
        yv0.v = _mm512_setzero_pd();
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy );                               // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy );                               // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy );                               // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy );                               // yv0[3] = y_buf[3]
        yv0.d[4] = *( y_buf + 4 * incy );                               // yv0[4] = y_buf[4]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:4]
    }

    // Handles (a_buf[0:31, 0:4] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]

        // Load the input values from vector X.
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                    // a_vec[2] = a_buf[16:23, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 16 );                    // a_vec[3] = a_buf[16:23, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 16 );                    // a_vec[4] = a_buf[16:23, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[16:23, 3] * x_buf[16:23]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv2.v, rhov[4].v );    // rhov[4] += a_buf[16:23, 4] * x_buf[16:23]

        // Load the input values from vector X.
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                    // a_vec[2] = a_buf[24:31, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 24 );                    // a_vec[3] = a_buf[24:31, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 24 );                    // a_vec[4] = a_buf[24:31, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 3] * x_buf[24:31]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv3.v, rhov[4].v );    // rhov[4] += a_buf[24:31, 4] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        av[2] += 32;
        av[3] += 32;
        av[4] += 32;
        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:4] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] + 8 );                     // a_vec[4] = a_buf[8:15, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv1.v, rhov[4].v );    // rhov[4] += a_buf[8:15, 4] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;
        av[4] += 16;
        x_buf += 16;
        i     += 16;
    }

    // Handles (a_buf[0:7, 0:4] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]
        a_vec[4].v =  _mm512_loadu_pd( av[4] );                         // a_vec[4] = a_buf[0:7, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:7, 4] * x_buf[0:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;
        av[4] += 8;
        x_buf += 8;
        i     += 8;
    }

    // Handles fringe cases -> (a_buf[0:m_left, 0:4] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v =  _mm512_maskz_loadu_pd( m_mask, av[3] );           // a_vec[3] = a_buf[0:m_left, 3]
        a_vec[4].v =  _mm512_maskz_loadu_pd( m_mask, av[4] );           // a_vec[4] = a_buf[0:m_left, 4]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:m_left]
        rhov[4].v = _mm512_fmadd_pd( a_vec[4].v, xv0.v, rhov[4].v );    // rhov[4] += a_buf[0:m_left, 4] * x_buf[0:m_left]
    }

    // This section of code is used to find the sum of  values in 5 vectors (rhov[0:4]),
    // and store the reesult into the 5 elements of rho vector.
    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
    rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]
    rho.d[3] = _mm512_reduce_add_pd( rhov[3].v );                       // rho[3] = a_buf[3, 0:m] * x_buf[0:m]
    rho.d[4] = _mm512_reduce_add_pd( rhov[4].v );                       // rho[4] = a_buf[4, 0:m] * x_buf[0:m]

    // yv0 =  alpha * rho + [beta * y_buf[0:4]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm512_storeu_pd'
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );                 // y_buf[0:4] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
        *( y_buf + 2 * incy) = yv0.d[2];
        *( y_buf + 3 * incy) = yv0.d[3];

        *( y_buf + 4 * incy) = yv0.d[4];
    }
}

void  bli_dgemv_t_zen_int_mx4_avx512
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

    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    __mmask8 n_mask  = (1 << (4)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:3]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[4];                                                     // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t a_vec[4];                                                    // a_vec[i] --> a_buf[0:7, i]
    v8df_t rho;                                                         // rho  --> Result for (a_buf[:, 0:3] * x[:])

    // Set up pointers for 5 rows of A (columns of A^T).
    double *restrict av[4];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );
    rho.v = _mm512_setzero_pd();

    // Creating an array of pointers for 4 columns of matrix A
    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[:, 1]
    av[2] = a_buf + 2 * lda;                                           // av[2] = a_buf[:, 2]
    av[3] = a_buf + 3 * lda;                                           // av[3] = a_buf[:, 3]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:4]
    }
    else
    {
        yv0.v = _mm512_setzero_pd();
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy );                               // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy );                               // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy );                               // yv0[2] = y_buf[2]
        yv0.d[3] = *( y_buf + 3 * incy );                               // yv0[3] = y_buf[3]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:4]
    }

    // Handles (a_buf[0:31, 0:3] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

        // Load the input values from vector X.
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                    // a_vec[2] = a_buf[16:23, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 16 );                    // a_vec[3] = a_buf[16:23, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv2.v, rhov[3].v );    // rhov[3] += a_buf[16:23, 3] * x_buf[16:23]

        // Load the input values from vector X.
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                    // a_vec[2] = a_buf[24:31, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 24 );                    // a_vec[3] = a_buf[24:31, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );    // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );    // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 3] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        av[2] += 32;
        av[3] += 32;

        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:3] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] + 8 );                     // a_vec[3] = a_buf[8:15, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 3] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;
        av[3] += 16;

        x_buf += 16;
        i     += 16;
    }

    // Handles (a_buf[0:7, 0:3] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]
        a_vec[3].v =  _mm512_loadu_pd( av[3] );                         // a_vec[3] = a_buf[0:7, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:7, 3] * x_buf[0:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;
        av[3] += 8;

        x_buf += 8;
        i    += 8;
    }

    // Handles fringe cases -> (a_buf[0:m_left, 0:3] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]
        a_vec[3].v =  _mm512_maskz_loadu_pd( m_mask, av[3] );           // a_vec[3] = a_buf[0:m_left, 3]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv0.v, rhov[3].v );    // rhov[3] += a_buf[0:m_left, 3] * x_buf[0:m_left]
    }

    // This section of code is used to find the sum of  values in 4 vectors (rhov[0:3]),
    // and store the reesult into the 4 elements of rho vector.
    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
    rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]
    rho.d[3] = _mm512_reduce_add_pd( rhov[3].v );                       // rho[3] = a_buf[3, 0:m] * x_buf[0:m]

    // yv0 =  alpha * rho + [beta * y_buf[0:3]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm512_storeu_pd'
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );                 // y_buf[0:3] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
        *( y_buf + 2 * incy) = yv0.d[2];
        *( y_buf + 3 * incy) = yv0.d[3];
    }
}

void  bli_dgemv_t_zen_int_mx3_avx512
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

    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    __mmask8 n_mask  = (1 << (3)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                             // yv0 --> y_buf[0:2]
    v8df_t xv0, xv1, xv2, xv3;                                              // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[3];                                                         // rhov[i] --> Accumulator for (a_buf[:, i] * x_buf[:])
    v8df_t a_vec[3];                                                        // a_vec[i] --> a_buf[0:7, i]
    v8df_t rho;                                                             // rho  --> Result for (a_buf[:, 0:2] * x[:])

    // Set up pointers for 5 rows of A (columns of A^T).
    double *restrict av[3];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );
    rho.v = _mm512_setzero_pd();

    // Creating an array of pointers for 3 columns of matrix A
    av[0] = a_buf + 0 * lda;                                               // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;                                               // av[1] = a_buf[:, 1]
    av[2] = a_buf + 2 * lda;                                               // av[2] = a_buf[:, 2]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:4]
    }
    else
    {
        yv0.v = _mm512_setzero_pd();
        // In case of non-unit stride y,
        // The inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy );                                   // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy );                                   // yv0[1] = y_buf[1]
        yv0.d[2] = *( y_buf + 2 * incy );                                   // yv0[2] = y_buf[2]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                            // yv0 = beta * yv0 = beta * y_buf[0:4]
    }

    // Handles (a_buf[0:31, 0:2] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                                  // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                             // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                             // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                             // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );        // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );        // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );        // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                              // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                         // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                         // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                         // a_vec[2] = a_buf[8:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );        // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );        // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );        // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]

        // Load the input values from vector X.
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                             // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                        // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                        // a_vec[1] = a_buf[16:23, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 16 );                        // a_vec[2] = a_buf[16:23, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );        // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );        // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );        // rhov[2] += a_buf[16:23, 2] * x_buf[16:23]

        // Load the input values from vector X.
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                             // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 24 );                        // a_vec[0] = a_buf[24:31, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 24 );                        // a_vec[1] = a_buf[24:31, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 24 );                        // a_vec[2] = a_buf[24:31, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv3.v, rhov[0].v );        // rhov[0] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv3.v, rhov[1].v );        // rhov[1] += a_buf[24:31, 1] * x_buf[24:31]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );        // rhov[2] += a_buf[24:31, 2] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        av[2] += 32;

        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:2] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Load the input values from vector X.
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] + 8 );                     // a_vec[2] = a_buf[8:15, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv1.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 2] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        av[2] += 16;

        x_buf += 16;
        i     += 16;
    }

    // Handles ( a_buf[0:7, 0:2] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]
        a_vec[2].v =  _mm512_loadu_pd( av[2] );                         // a_vec[2] = a_buf[0:7, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:7, 1] * x_buf[0:7]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:7, 2] * x_buf[0:7]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        av[2] += 8;

        x_buf += 8;
        i    += 8;
    }

    // Handles fringe cases -> (a_buf[0:m_left, 0:2] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]
        a_vec[2].v =  _mm512_maskz_loadu_pd( m_mask, av[2] );           // a_vec[2] = a_buf[0:m_left, 2]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv0.v, rhov[2].v );    // rhov[2] += a_buf[0:m_left, 2] * x_buf[0:m_left]
    }

    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]
    rho.d[2] = _mm512_reduce_add_pd( rhov[2].v );                       // rho[2] = a_buf[2, 0:m] * x_buf[0:m]

    // yv0 =  alpha * rho + [beta * y_buf[0:2]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        // In case of unit stride y,
        // The result is moved back onto vector y using '_mm512_storeu_pd'
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );                 // y_buf[0:2] = yv0
    }
    else
    {
        // In case of non-unit stride y,
        // The result is manually moved back onto vector y
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
        *( y_buf + 2 * incy) = yv0.d[2];
    }
}

void  bli_dgemv_t_zen_int_mx2_avx512
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

    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    __mmask8 n_mask  = (1 << (2)) - 1;

    // vector variables used load inputs.
    v8df_t yv0;                                                         // yv0 --> y_buf[0:1]
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[4];                                                     // rhov[0] & rhov[2] --> Accumulator for (a_buf[:, 0] * x_buf[:])    rhov[1] & rhov[3] --> Accumulator for (a_buf[:, 1] * x_buf[:])
    v8df_t a_vec[4];                                                    // a_vec[0] --> a_buf[0:7, 0]    a_vec[1] --> a_buf[0:7, 1]    a_vec[2] --> a_buf[8:15, 0]    a_vec[0] --> a_buf[8:15, 1]
    v8df_t rho;                                                         // rho  --> Result for (a_buf[:, 0:1] * x[:])

    // Set up pointers for 5 rows of A (columns of A^T).
    double *restrict av[2];
    v8df_t alphav;
    v8df_t betav;

    alphav.v = _mm512_set1_pd( *alpha );
    betav.v = _mm512_set1_pd( *beta );
    rho.v = _mm512_setzero_pd();

    // Creating an array of pointers for 2 columns of matrix A
    av[0] = a_buf + 0 * lda;                                           // av[0] = a_buf[:, 0]
    av[1] = a_buf + 1 * lda;                                           // av[1] = a_buf[:, 1]

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv0.v = _mm512_setzero_pd();
    }
    else if ( incy == 1)
    {
        // In case of unit stride y, the inputs on vector are moved to register yv using masked vector load
        // Followed by '_mm512_mul_pd' to get beta * y
        yv0.v = _mm512_mul_pd( betav.v, _mm512_maskz_loadu_pd( n_mask, y_buf ) );     // yv0 = beta * y_buf[0:1]
    }
    else
    {
        // In case of non-unit stride y,
        // The 3 inputs on vector y are manually moved to registor yv0
        yv0.d[0] = *( y_buf + 0 * incy);                                // yv0[0] = y_buf[0]
        yv0.d[1] = *( y_buf + 1 * incy);                                // yv0[1] = y_buf[1]

        yv0.v = _mm512_mul_pd( betav.v, yv0.v );                        // yv0 = beta * yv0 = beta * y_buf[0:1]
    }

    // Handles (a_buf[0:31, 0:1] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]

        // Load the input values from vector X
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[3].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 1] * x_buf[8:15]

        // Load the input values from vector X
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] + 16 );                    // a_vec[0] = a_buf[16:23, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] + 16 );                    // a_vec[1] = a_buf[16:23, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv2.v, rhov[0].v );    // rhov[0] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv2.v, rhov[1].v );    // rhov[1] += a_buf[16:23, 1] * x_buf[16:23]

        // Load the input values from vector X
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm512_loadu_pd( av[0] + 24 );                    // a_vec[0] = a_buf[24:31, 0]
        a_vec[3].v =  _mm512_loadu_pd( av[1] + 24 );                    // a_vec[1] = a_buf[24:31, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv3.v, rhov[2].v );    // rhov[2] += a_buf[24:31, 0] * x_buf[24:31]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 1] * x_buf[24:31]

        // Incrementing pointers
        av[0] += 32;
        av[1] += 32;
        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0:1] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]

        // Load the input values from vector X
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[2].v =  _mm512_loadu_pd( av[0] + 8 );                     // a_vec[0] = a_buf[8:15, 0]
        a_vec[3].v =  _mm512_loadu_pd( av[1] + 8 );                     // a_vec[1] = a_buf[8:15, 1]

        // perform: rho?v += a?v * x0v;
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv1.v, rhov[2].v );    // rhov[2] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv1.v, rhov[3].v );    // rhov[3] += a_buf[8:15, 1] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 16;
        av[1] += 16;
        x_buf += 16;
        i    += 16;
    }

    rhov[0].v = _mm512_add_pd( rhov[2].v, rhov[0].v );                  // rhov[0] = rhov[0] + rhov[2]
    rhov[1].v = _mm512_add_pd( rhov[3].v, rhov[1].v );                  // rhov[1] = rhov[1] + rhov[3]

    // Handles (a_buf[0:7, 0:1] * x_buf[0:7])
    if ( (i + 7) < m )
    {
        // Load the input values from vector X
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( av[0] );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( av[1] );                         // a_vec[1] = a_buf[0:7, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 1] * x_buf[8:15]

        // Incrementing pointers
        av[0] += 8;
        av[1] += 8;
        x_buf += 8;
        i    += 8;
    }

    // Handling fringe cases -> (a_buf[0:m_left, 0:1] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_maskz_loadu_pd( m_mask, av[0] );           // a_vec[0] = a_buf[0:m_left, 0]
        a_vec[1].v =  _mm512_maskz_loadu_pd( m_mask, av[1] );           // a_vec[1] = a_buf[0:m_left, 1]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv0.v, rhov[1].v );    // rhov[1] += a_buf[0:m_left, 1] * x_buf[0:m_left]
    }

    rho.d[0] = _mm512_reduce_add_pd( rhov[0].v );                       // rho[0] = a_buf[0, 0:m] * x_buf[0:m]
    rho.d[1] = _mm512_reduce_add_pd( rhov[1].v );                       // rho[1] = a_buf[1, 0:m] * x_buf[0:m]

    // yv0[0:1] =  (alpha * rho) + [beta * y_buf[0:1]]
    yv0.v = _mm512_fmadd_pd( alphav.v, rho.v, yv0.v );

    // Store the result back into vector y
    if ( incy == 1)
    {
        _mm512_mask_storeu_pd( y_buf , n_mask, yv0.v );
    }
    else
    {
        *( y_buf + 0 * incy) = yv0.d[0];
        *( y_buf + 1 * incy) = yv0.d[1];
    }
}

void  bli_dgemv_t_zen_int_mx1_avx512
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

    dim_t m_left = m % 8;

    // i denotes the number of columns completed
    dim_t i = 0;

    // Mask used to load x, when fringe cases are considered.
    __mmask8 m_mask  = (1 << (m_left)) - 1;

    // vector variables used load inputs.
    v8df_t xv0, xv1, xv2, xv3;                                          // xv0 --> x_buf[0:7], xv1 --> x_buf[8:15], xv2 --> x_buf[16:23], xv3 --> x_buf[24:31]
    v8df_t rhov[4];                                                     // rhov[0:3]--> Accumulator for (a_buf[:, 0] * x_buf[:])
    v8df_t a_vec[4];                                                    // a_vec[0] --> a_buf[0:7, 0]    a_vec[1] --> a_buf[8:15, 0]    a_vec[2] --> a_buf[16:23, 0]    a_vec[3] --> a_buf[24:31, 0]

    double yv, rh;                                                      // yv  -->   y_buf[0]    rh  --> Result for (a_buf[:, 0] * x[:])

    // Clearing vectors before use
    rhov[0].v = _mm512_setzero_pd();
    rhov[1].v = _mm512_setzero_pd();
    rhov[2].v = _mm512_setzero_pd();
    rhov[3].v = _mm512_setzero_pd();

    // Loading data from vector y
    if (bli_deq0( *beta ))
    {
        // yv is assigned zero if beta is 0
        yv = 0;
    }
    else
    {
        // yv = beta * y_buf[0]
        yv = (*beta) * (*y_buf);
    }

    // Handles (a_buf[0:31, 0] * x_buf[0:31])
    for ( i = 0; (i + 31) < m; i += 32 )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]
        xv2.v =  _mm512_loadu_pd( x_buf + 16 );                         // xv2 = x_buf[16:23]
        xv3.v =  _mm512_loadu_pd( x_buf + 24 );                         // xv3 = x_buf[24:31]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( a_buf + 8 );                     // a_vec[1] = a_buf[8:15, 0]
        a_vec[2].v =  _mm512_loadu_pd( a_buf + 16 );                    // a_vec[2] = a_buf[16:23, 0]
        a_vec[3].v =  _mm512_loadu_pd( a_buf + 24 );                    // a_vec[3] = a_buf[24:31, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 0] * x_buf[8:15]
        rhov[2].v = _mm512_fmadd_pd( a_vec[2].v, xv2.v, rhov[2].v );    // rhov[2] += a_buf[16:23, 0] * x_buf[16:23]
        rhov[3].v = _mm512_fmadd_pd( a_vec[3].v, xv3.v, rhov[3].v );    // rhov[3] += a_buf[24:31, 0] * x_buf[24:31]

        // Incrementing pointers
        a_buf += 32;
        x_buf += 32;
    }

    // Handles (a_buf[0:15, 0] * x_buf[0:15])
    if ( (i + 15) < m )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]
        xv1.v =  _mm512_loadu_pd( x_buf + 8 );                          // xv1 = x_buf[8:15]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:7, 0]
        a_vec[1].v =  _mm512_loadu_pd( a_buf + 8 );                     // a_vec[1] = a_buf[8:15, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]
        rhov[1].v = _mm512_fmadd_pd( a_vec[1].v, xv1.v, rhov[1].v );    // rhov[1] += a_buf[8:15, 0] * x_buf[8:15]

        // Incrementing pointers
        a_buf += 16;
        x_buf += 16;
        i    += 16;
    }

    rhov[0].v = _mm512_add_pd(rhov[0].v, rhov[2].v);                    // rhov[0] = rhov[0] + rhov[2]
    rhov[1].v = _mm512_add_pd(rhov[1].v, rhov[3].v);                    // rhsov[1] = rhov[1] + rhov[3]

    rhov[0].v = _mm512_add_pd(rhov[0].v, rhov[1].v);                    // rhov[0] = rhov[0] + rhov[1] => rhov[0] + rhov[1] + rhov[2] + rhov[3]

    // Handles (a_buf[0:7, 0] * x_buf[0:7])
    if (i + 7 < m)
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_loadu_pd( x_buf );                              // xv0 = x_buf[0:7]

        // Load the input values from Matrix A
        a_vec[0].v =  _mm512_loadu_pd( a_buf );                         // a_vec[0] = a_buf[0:7, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v );    // rhov[0] += a_buf[0:7, 0] * x_buf[0:7]

        // Incrementing pointers
        a_buf += 8;
        x_buf += 8;
        i    += 8;
    }

    // Handles fringe cases -> (a_buf[0:m_left, 0] * x_buf[0:m_left])
    if( m_left )
    {
        // Load the input values from vector X.
        xv0.v =  _mm512_maskz_loadu_pd( m_mask, x_buf );                // xv0 = x_buf[0:m_left]

        // Load the input values from Matrix A
        a_vec[0].v  =  _mm512_maskz_loadu_pd( m_mask, a_buf );          // a_vec[0] = a_buf[0:m_left, 0]

        // perform: rho?v += a?v * x0v;
        rhov[0].v    = _mm512_fmadd_pd( a_vec[0].v, xv0.v, rhov[0].v ); // rhov[0] += a_buf[0:m_left, 0] * x_buf[0:m_left]
    }

    rh = _mm512_reduce_add_pd( rhov[0].v );                             // rh = a_buf[0, 0:m] * x_buf[0:m]

    // yv =  alpha * rh + [beta * y_buf[0]]
    yv += ((*alpha) * rh);

    // Store the result back into vector y
    *( y_buf ) = yv;
}
