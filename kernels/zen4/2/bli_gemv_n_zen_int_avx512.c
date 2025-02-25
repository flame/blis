/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * GEMV Operation (assuming op(A) = NO_TRANSPOSE):
 *      y := beta * y + alpha * op(A) * x
 *  where,
 *      y - m-dimensional vector when op(A) = NO_TRANSPOSE.
 *      x - n-dimensional vector when op(A) = NO_TRANSPOSE.
 *      A - m x n dimensional matrix.
 *      alpha, beta - scalars.
 */

// Function pointers for n-biased kernels.
static dgemv_ker_ft n_ker_fp[5][4] =
{
    { bli_dgemv_n_zen_int_32x8n_avx512,
      bli_dgemv_n_zen_int_16x8n_avx512,
      bli_dgemv_n_zen_int_8x8n_avx512,
      bli_dgemv_n_zen_int_m_leftx8n_avx512 },
    { bli_dgemv_n_zen_int_32x1n_avx512,
      bli_dgemv_n_zen_int_16x1n_avx512,
      bli_dgemv_n_zen_int_8x1n_avx512,
      bli_dgemv_n_zen_int_m_leftx1n_avx512 },
    { bli_dgemv_n_zen_int_32x2n_avx512,
      bli_dgemv_n_zen_int_16x2n_avx512,
      bli_dgemv_n_zen_int_8x2n_avx512,
      bli_dgemv_n_zen_int_m_leftx2n_avx512 },
    { bli_dgemv_n_zen_int_32x3n_avx512,
      bli_dgemv_n_zen_int_16x3n_avx512,
      bli_dgemv_n_zen_int_8x3n_avx512,
      bli_dgemv_n_zen_int_m_leftx3n_avx512 },
    { bli_dgemv_n_zen_int_32x4n_avx512,
      bli_dgemv_n_zen_int_16x4n_avx512,
      bli_dgemv_n_zen_int_8x4n_avx512,
      bli_dgemv_n_zen_int_m_leftx4n_avx512 }
};

// Function pointers for m-biased kernels.
static dgemv_ker_ft m_ker_fp[8] =
{
    bli_dgemv_n_zen_int_16mx8_avx512,   // base kernel
    bli_dgemv_n_zen_int_16mx1_avx512,   // n = 1
    bli_dgemv_n_zen_int_16mx2_avx512,   // n = 2
    bli_dgemv_n_zen_int_16mx3_avx512,   // n = 3
    bli_dgemv_n_zen_int_16mx4_avx512,   // n = 4
    bli_dgemv_n_zen_int_16mx5_avx512,   // n = 5
    bli_dgemv_n_zen_int_16mx6_avx512,   // n = 6
    bli_dgemv_n_zen_int_16mx7_avx512    // n = 7
};


/**
 * bli_dgemv_n_avx512(...) handles cases where op(A) = NO_TRANSPOSE and invokes
 * the n/m-biased kernel based on the conditions satisfied.
 */
void bli_dgemv_n_avx512
     (
       trans_t transa,
       conj_t  conjx,
       dim_t   m,
       dim_t   n,
       double* alpha,
       double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx,
       double* beta,
       double* y, inc_t incy,
       cntx_t* cntx
     )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_4 );

    // Invoking the reference kernel to handle general stride.
    if ( ( rs_a != 1 ) && ( cs_a != 1 ) )
    {
        bli_dgemv_zen_ref
        (
          transa,
          m,
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          NULL
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    dim_t   m0, n0;
    inc_t   rs_at, cs_at;
    conj_t  conja;

    // Memory pool declarations for packing vector Y.
    mem_t   mem_bufY;
    rntm_t  rntm;
    double* y_temp    = y;
    inc_t   temp_incy = incy;

    // Boolean to check if y vector is packed and memory needs to be freed.
    bool is_y_temp_buf_created = FALSE;

    // Update dimensions and strides based on op(A).
    bli_set_dims_incs_with_trans( transa,
                                  m, n, rs_a, cs_a,
                                  &m0, &n0, &rs_at, &cs_at );

    conja = bli_extract_conj( transa );

    // Function pointer declaration for the functions that will be used.
    dscal2v_ker_ft scal2v_kr_ptr;       // DSCAL2V
    dscalv_ker_ft  scalv_kr_ptr;        // DSCALV
    dcopyv_ker_ft  copyv_kr_ptr;        // DCOPYV

    scal2v_kr_ptr = bli_dscal2v_zen_int_avx512;     // DSCAL2V
    scalv_kr_ptr  = bli_dscalv_zen_int_avx512;      // DSCALV
    copyv_kr_ptr  = bli_dcopyv_zen4_asm_avx512;     // DCOPYV

    // If n0 is less than 5 and y is unit strided, invoke optimized n-biased
    // kernels which handle beta scaling of y vector.
    if ( n0 < 5 && incy == 1 )
    {
        dim_t m_idx = 0;    // 32x8n kernel
        if      ( 32 > m0 && m0 > 15 ) m_idx = 1;   // 16x8n kernel; [16,31)
        else if ( 16 > m0 && m0 > 7  ) m_idx = 2;   // 8x8n kernel; [8,15)
        else if (  8 > m0 && m0 > 0  ) m_idx = 3;   // m_leftx8n kernel; [1,7)

        // Invoke respective kernel based on n dimension.
        n_ker_fp[n0][m_idx]
        (
          conja,
          conjx,
          m0,
          n0,
          alpha,
          a, rs_at, cs_at,
          x, incx,
          beta,
          y_temp, temp_incy,
          cntx
        );

        return;
    }

    /**
     * If y has non-unit increments and alpha is non-zero, y is packed and
     * scaled by beta. The scaled contents are copied to a temp buffer (y_temp)
     * and passed to the kernels. At the end, the contents of y_temp are copied
     * back to y and memory is freed.
     * If alpha is zero, the GEMV operation is reduced to y := beta * y, thus,
     * packing of y is unnecessary so y is only scaled by beta and returned.
     */
    if ( (incy != 1) && (!bli_deq0( *alpha )))
    {
        /**
         * Initialize mem pool buffer to NULL and size to 0.
         * "buf" and "size" fields are assigned once memory is allocated from
         * the pool in bli_pba_acquire_m().
         * This will ensure bli_mem_is_alloc() will be passed on an allocated
         * memory if created or a NULL .
         */
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;

        /**
         * In order to get the buffer from pool via rntm access to memory broker
         * is needed. Following are initializations for rntm.
         */
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        // Calculate the size required for m0 double elements in vector Y.
        size_t buffer_size = m0 * sizeof( double );

#ifdef BLIS_ENABLE_MEM_TRACING
        printf("bli_dgemv_n_avx512(): get mem pool block\n");
#endif

        /**
         * Acquire a Buffer(m0*size(double)) from the memory broker and save the
         * associated mem_t entry to mem_bufY.
         */
        bli_pba_acquire_m
        (
          &rntm,
          buffer_size,
          BLIS_BUFFER_FOR_B_PANEL,
          &mem_bufY
        );

        // Continue packing Y if buffer memory is allocated.
        if ( bli_mem_is_alloc( &mem_bufY ) )
        {
            y_temp = bli_mem_buffer( &mem_bufY );

            // Using unit-stride for y_temp vector.
            temp_incy = 1;

            // Invoke the SCAL2V function using the function pointer.
            scal2v_kr_ptr
            (
              BLIS_NO_CONJUGATE,
              m0,
              beta,
              y, incy,
              y_temp, temp_incy,
              cntx
            );

            /**
             * Set y is packed as the memory allocation was successful
             * and contents have been scaled and copied to a temp buffer.
             */
            is_y_temp_buf_created = TRUE;
        }
    }
    else
    {
        // Invoke the SCALV function using the function pointer
        scalv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          m0,
          beta,
          y_temp, temp_incy,
          cntx
        );
    }

    /**
     * If alpha is zero, the GEMV operation is reduced to y := beta * y, thus,
     * y is only scaled by beta and returned.
     */
    if( bli_deq0( *alpha ) )
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    /**
     * Invoking the m-biased kernels.
     * If n0 is less than 8, we directly invoke the respective fringe kernels,
     * else the base kernel is invoked.
     */
    dim_t n_idx = 0;
    if ( n0 < 8 ) n_idx = n0;

    // Invoke respective m_kernel
    m_ker_fp[n_idx]
    (
      conja,
      conjx,
      m0,
      n0,
      alpha,
      a, rs_at, cs_at,
      x, incx,
      beta,
      y_temp, temp_incy,
      cntx
    );

    // If y was packed into y_temp, copy the contents back to y and free memory.
    if (is_y_temp_buf_created)
    {
        /**
         * Invoke COPYV to store the result from unit-strided y_buf to non-unit
         * strided y.
         */
        copyv_kr_ptr
        (
          BLIS_NO_CONJUGATE,
          m0,
          y_temp, temp_incy,
          y, incy,
          cntx
        );

#ifdef BLIS_ENABLE_MEM_TRACING
        printf( "bli_dgemv_n_avx512(): releasing mem pool block\n" );
#endif
        // Return the buffer to pool.
        bli_pba_release( &rntm , &mem_bufY );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_4 );
}

/**
 * ----------------------------------------------------------------------------
 * -------------------------- m-biased DGEMV kernels --------------------------
 * ----------------------------------------------------------------------------
 * - The m-biased kernels iterate over block of size 16x8 over the m-dimension
 *   keeping the n-dimension(8) fixed to calculate the intermediate result for
 *   the entire y vector, while simulataneously updating y with the intermediate
 *   results.
 * - Since these kernels calculate intermeidate y vector they do not handle y
 *   scaling by beta within the kernel hence, y is scaled by beta before
 *   invoking the kernel.
 * - The kernels follow the below naming scheme:
 *   bli_dgemv_n_zen_int_<M_BLOCK>mx<N_BLOCK>_avx512
 *   where, dgemv_n denotes NO_TRANSPOSE implementation.
 * - The main kernel, bli_dgemv_n_zen_int_16mx8_avx512(...), handles the DGEMV
 *   operation by breaking A matrix into blocks of size 16x8, i.e., 16 rows and
 *   8 columns.
 *
 *                                        i=0    i=1    ...
 *      |----|          |----|         |------|------|-------|     |---|
 *      |    |          |    |         |      |      |       |     | 8 | i=0
 *  j=0 | 16 |      j=0 | 16 |     j=0 | 16x8 | 16x8 |       |     |---|
 *      |    |          |    |         |      |      |       |     | 8 | i=1
 *      |----|          |----|         |------|------|       |     |---|
 *      |    |  :=      |    |  +      |      |      |       |  +  |   | ...
 *  j=1 | 16 |      j=1 | 16 |     j=1 | 16x8 | 16x8 |       |     |   |
 *      |----|          |----|         |      |      |       |     |   |
 *      |    |          |    |         |------|------|       |     |   |
 *  ... |    |      ... |    |     ... |      |      |       |     |   |
 *      |    |          |    |         |  mx8 |  mx8 |       |     |   |
 *      |----|          |----|         |------|------|-------|     |---|
 *        y               y                      A                   x
 *
 * y vector:
 * y0, y1, y2, y3, ...
 *
 * A matrix:
 * a00, a01, a02, a03, ...
 * a10, a11, a12, a13, ...
 * a20, a21, a22, a23, ...
 * a30, a31, a32, a33, ...
 * ...
 *
 * x vector:
 * x0, x1, x2, x3, ...
 *
 * Outer Loop (i-loop) over n-dimension:
 *     - Scale and broadcast 8 elements from x vector.
 *       zmm0 = alpha*x0, alpha*x0, alpha*x0, alpha*x0, ...     // zmm0 = alpha*x[0*incx]
 *       zmm1 = alpha*x1, alpha*x1, alpha*x1, alpha*x1, ...     // zmm1 = alpha*x[1*incx]
 *       ...
 *       zmm7 = alpha*x7, alpha*x7, alpha*x7, alpha*x7, ...     // zmm7 = alpha*x[7*incx]
 *
 *     - Inner Loop (j-loop) over m-dimension:
 *         - Load 16 elements from y vector.
 *           zmm8 = y0, y1,  y2,  y3, ...                       // zmm8 = ybuf[ 0:7]
 *           zmm9 = y8, y9, y10, y11, ...                       // zmm9 = ybuf[8:15]
 *
 *         - Load 16 rows & 8 columns from A matrix.
 *           zmm10 = a(0,0), a(1,0),  a(2,0),  a(3,0), ...      // zmm10 = abuf[ 0:7, 0]
 *           zmm11 = a(8,0), a(9,0), a(10,0), a(11,0), ...      // zmm11 = abuf[8:15, 0]
 *           ...
 *           zmm24 = a(0,7), a(1,7),  a(2,7),  a(3,7), ...      // zmm24 = abuf[ 0:7, 7]
 *           zmm25 = a(8,7), a(9,7), a(10,7), a(11,7), ...      // zmm11 = abuf[8:15, 7]
 *
 *         - Perform the operation, y := y + A * (alpha * x) and store the
 *           result in intermediate registers.
 *           zmm8 = zmm8 + zmm10 * zmm0                         // ybuf[ 0:7] += abuf[ 0:7, 0] * (alpha*x[0])
 *           zmm8 = zmm8 + zmm12 * zmm1                         // ybuf[ 0:7] += abuf[ 0:7, 1] * (alpha*x[1])
 *           ...
 *           zmm8 = zmm8 + zmm22 * zmm6                         // ybuf[ 0:7] += abuf[ 0:7, 6] * (alpha*x[6])
 *           zmm8 = zmm8 + zmm24 * zmm7                         // ybuf[ 0:7] += abuf[ 0:7, 7] * (alpha*x[7])
 *
 *           zmm9 = zmm9 + zmm11 * zmm0                         // ybuf[8:15] += abuf[8:15, 0] * (alpha*x[0])
 *           zmm9 = zmm9 + zmm13 * zmm1                         // ybuf[8:15] += abuf[8:15, 1] * (alpha*x[1])
 *           ...
 *           zmm9 = zmm9 + zmm23 * zmm6                         // ybuf[8:15] += abuf[8:15, 6] * (alpha*x[6])
 *           zmm9 = zmm9 + zmm25 * zmm7                         // ybuf[8:15] += abuf[8:15, 7] * (alpha*x[7])
 *
 *         - Store the intermediate result back to y buffer.
 *            ybuf[ 0:7] = zmm8
 *            ybuf[8:15] = zmm9
 *
 *         - Update ybuf to point to the next block of 16 elements of y.
 *           ybuf += 16*incy;
 *
 *         - Move abuf pointer to the next block of 16 rows.
 *           abuf += 16*rs_a;
 *
 *     - Update xbuf to point to the next block of 8 elements of x.
 *       xbuf += 8;
 */
void bli_dgemv_n_zen_int_16mx8_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i, j;

    /**
     * m16_iter: Number of iterations while handling 16 rows at once.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows after handling blocks of 16xN and 8xN.
     */
    dim_t m16_iter = m / 16;
    dim_t m_left   = m % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    /**
     * n8_iter: Number of iterations while handling 8 columns at once.
     * n_left : Number of leftover columns.
     */
    dim_t n8_iter = n / 8;
    dim_t n_left  = n % 8;

    // x vector
    __m512d zmm0, zmm1, zmm2, zmm3;
    __m512d zmm4, zmm5, zmm6, zmm7;

    // y vector
    __m512d zmm8, zmm9;

    // A matrix
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;
    __m512d zmm18, zmm19, zmm20, zmm21;
    __m512d zmm22, zmm23, zmm24, zmm25;

    // Outer loop over the n dimension, i.e., columns of A matrix and elements
    // in x vector.
    for ( i = 0; i < n8_iter; ++i )
    {
        // Initialize ybuf to the beginning of y vector for every iteration.
        ybuf = y;
        // Move abuf to the beginning of the next block of columns.
        abuf = a + 8*i*cs_a;

        // Scale by alpha and broadcast 8 elements from x vector.
        zmm0 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );   // zmm0 = alpha*x[0*incx]
        zmm1 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );   // zmm1 = alpha*x[1*incx]
        zmm2 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );   // zmm2 = alpha*x[2*incx]
        zmm3 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );   // zmm3 = alpha*x[3*incx]
        zmm4 = _mm512_set1_pd( *alpha * (*(xbuf + 4*incx)) );   // zmm4 = alpha*x[4*incx]
        zmm5 = _mm512_set1_pd( *alpha * (*(xbuf + 5*incx)) );   // zmm5 = alpha*x[5*incx]
        zmm6 = _mm512_set1_pd( *alpha * (*(xbuf + 6*incx)) );   // zmm6 = alpha*x[6*incx]
        zmm7 = _mm512_set1_pd( *alpha * (*(xbuf + 7*incx)) );   // zmm7 = alpha*x[7*incx]

        // Inner loop over the m dimension, i.e., rows of A matrix and elements
        // in y vector.
        // This loop handles A matrix in blocks of 16x8, calculates the
        // intermediate results and stores it back into y.
        for ( j = 0; j < m16_iter; ++j )
        {
            // Load 16 elements from y.
            zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[ 0:7]
            zmm9 = _mm512_loadu_pd( ybuf + 8 );                 // zmm9 = ybuf[8:15]

            // Load 16 rows and 8 columns from A.
            zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[ 0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf + 8*rs_a + 0*cs_a );  // zmm11 = abuf[8:15, 0]

            zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[ 0:7, 1]
            zmm13 = _mm512_loadu_pd( abuf + 8*rs_a + 1*cs_a );  // zmm13 = abuf[8:15, 1]

            zmm14 = _mm512_loadu_pd( abuf + 0*rs_a + 2*cs_a );  // zmm14 = abuf[ 0:7, 2]
            zmm15 = _mm512_loadu_pd( abuf + 8*rs_a + 2*cs_a );  // zmm15 = abuf[8:15, 2]

            zmm16 = _mm512_loadu_pd( abuf + 0*rs_a + 3*cs_a );  // zmm16 = abuf[ 0:7, 3]
            zmm17 = _mm512_loadu_pd( abuf + 8*rs_a + 3*cs_a );  // zmm17 = abuf[8:15, 3]

            zmm18 = _mm512_loadu_pd( abuf + 0*rs_a + 4*cs_a );  // zmm18 = abuf[ 0:7, 4]
            zmm19 = _mm512_loadu_pd( abuf + 8*rs_a + 4*cs_a );  // zmm19 = abuf[8:15, 4]

            zmm20 = _mm512_loadu_pd( abuf + 0*rs_a + 5*cs_a );  // zmm20 = abuf[ 0:7, 5]
            zmm21 = _mm512_loadu_pd( abuf + 8*rs_a + 5*cs_a );  // zmm21 = abuf[8:15, 5]

            zmm22 = _mm512_loadu_pd( abuf + 0*rs_a + 6*cs_a );  // zmm22 = abuf[ 0:7, 6]
            zmm23 = _mm512_loadu_pd( abuf + 8*rs_a + 6*cs_a );  // zmm23 = abuf[8:15, 6]

            zmm24 = _mm512_loadu_pd( abuf + 0*rs_a + 7*cs_a );  // zmm24 = abuf[ 0:7, 7]
            zmm25 = _mm512_loadu_pd( abuf + 8*rs_a + 7*cs_a );  // zmm25 = abuf[8:15, 7]

            // Performing the operation y := y + A * (alpha * x) and storing the
            // result to intermediate registers.
            zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 0] * (alpha*x[0])
            zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 1] * (alpha*x[1])
            zmm8 = _mm512_fmadd_pd( zmm14, zmm2, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 2] * (alpha*x[2])
            zmm8 = _mm512_fmadd_pd( zmm16, zmm3, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 3] * (alpha*x[3])
            zmm8 = _mm512_fmadd_pd( zmm18, zmm4, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 4] * (alpha*x[4])
            zmm8 = _mm512_fmadd_pd( zmm20, zmm5, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 5] * (alpha*x[5])
            zmm8 = _mm512_fmadd_pd( zmm22, zmm6, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 6] * (alpha*x[6])
            zmm8 = _mm512_fmadd_pd( zmm24, zmm7, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 7] * (alpha*x[7])

            zmm9 = _mm512_fmadd_pd( zmm11, zmm0, zmm9 );        // ybuf[8:15] += abuf[8:15, 0] * (alpha*x[0])
            zmm9 = _mm512_fmadd_pd( zmm13, zmm1, zmm9 );        // ybuf[8:15] += abuf[8:15, 1] * (alpha*x[1])
            zmm9 = _mm512_fmadd_pd( zmm15, zmm2, zmm9 );        // ybuf[8:15] += abuf[8:15, 2] * (alpha*x[2])
            zmm9 = _mm512_fmadd_pd( zmm17, zmm3, zmm9 );        // ybuf[8:15] += abuf[8:15, 3] * (alpha*x[3])
            zmm9 = _mm512_fmadd_pd( zmm19, zmm4, zmm9 );        // ybuf[8:15] += abuf[8:15, 4] * (alpha*x[4])
            zmm9 = _mm512_fmadd_pd( zmm21, zmm5, zmm9 );        // ybuf[8:15] += abuf[8:15, 5] * (alpha*x[5])
            zmm9 = _mm512_fmadd_pd( zmm23, zmm6, zmm9 );        // ybuf[8:15] += abuf[8:15, 6] * (alpha*x[6])
            zmm9 = _mm512_fmadd_pd( zmm25, zmm7, zmm9 );        // ybuf[8:15] += abuf[8:15, 7] * (alpha*x[7])

            // Store the intermediate result back to the y buffer.
            _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[ 0:7] = zmm8
            _mm512_storeu_pd( ybuf + 8*incy, zmm9 );            // ybuf[8:15] = zmm9

            ybuf += 16*incy;    // Move ybuf to the next block of 16 elements.
            abuf += 16*rs_a;    // Move abuf to the next block of 16 rows.
        }

        // If m8_iter is non-zero, handle the remaining 8 rows.
        if ( m8_iter )
        {
            // Load 8 elements from y.
            zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[0:7]

            // Load 8 rows and 8 columns from A.
            zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[0:7, 0]

            zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[0:7, 1]

            zmm14 = _mm512_loadu_pd( abuf + 0*rs_a + 2*cs_a );  // zmm14 = abuf[0:7, 2]

            zmm16 = _mm512_loadu_pd( abuf + 0*rs_a + 3*cs_a );  // zmm16 = abuf[0:7, 3]

            zmm18 = _mm512_loadu_pd( abuf + 0*rs_a + 4*cs_a );  // zmm18 = abuf[0:7, 4]

            zmm20 = _mm512_loadu_pd( abuf + 0*rs_a + 5*cs_a );  // zmm20 = abuf[0:7, 5]

            zmm22 = _mm512_loadu_pd( abuf + 0*rs_a + 6*cs_a );  // zmm22 = abuf[0:7, 6]

            zmm24 = _mm512_loadu_pd( abuf + 0*rs_a + 7*cs_a );  // zmm24 = abuf[0:7, 7]

            // Performing the operation y := y + A * (alpha * x) and storing the
            // result to intermediate registers.
            zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[0:7] += abuf[0:7, 0] * (alpha*x[0])
            zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[0:7] += abuf[0:7, 1] * (alpha*x[1])
            zmm8 = _mm512_fmadd_pd( zmm14, zmm2, zmm8 );        // ybuf[0:7] += abuf[0:7, 2] * (alpha*x[2])
            zmm8 = _mm512_fmadd_pd( zmm16, zmm3, zmm8 );        // ybuf[0:7] += abuf[0:7, 3] * (alpha*x[3])
            zmm8 = _mm512_fmadd_pd( zmm18, zmm4, zmm8 );        // ybuf[0:7] += abuf[0:7, 4] * (alpha*x[4])
            zmm8 = _mm512_fmadd_pd( zmm20, zmm5, zmm8 );        // ybuf[0:7] += abuf[0:7, 5] * (alpha*x[5])
            zmm8 = _mm512_fmadd_pd( zmm22, zmm6, zmm8 );        // ybuf[0:7] += abuf[0:7, 6] * (alpha*x[6])
            zmm8 = _mm512_fmadd_pd( zmm24, zmm7, zmm8 );        // ybuf[0:7] += abuf[0:7, 7] * (alpha*x[7])

            // Store the intermediate result back to the y buffer.
            _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[0:7] = zmm8

            ybuf += 8*incy;    // Move ybuf to the next block of 8 elements.
            abuf += 8*rs_a;    // Move ybuf to the next block of 8 elements.
        }

        // If m_left is non-zero, handle the remaining rows using masked
        // operations.
        if ( m_left )
        {
            // Generate the mask based on the value of m_left.
            // Possible values of m_mask:
            // m_left = 0-7
            // m_mask = [2^0, 2^7] - 1 = [0, 127] = [0x00, 0x7f]
            // m_mask = [0b00000000, 0b01111111]
            __mmask8 m_mask = (1 << m_left) - 1;

            // Load m_left elements from y.
            zmm8 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );              // zmm8 = ybuf[0:m_left]

            // Load m_left rows and 8 columns from A.
            zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 0*cs_a );    // zmm10 = abuf[0:m_left, 0]

            zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 1*cs_a );    // zmm12 = abuf[0:m_left, 1]

            zmm14 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 2*cs_a );    // zmm14 = abuf[0:m_left, 2]

            zmm16 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 3*cs_a );    // zmm16 = abuf[0:m_left, 3]

            zmm18 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 4*cs_a );    // zmm18 = abuf[0:m_left, 4]

            zmm20 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 5*cs_a );    // zmm20 = abuf[0:m_left, 5]

            zmm22 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 6*cs_a );    // zmm22 = abuf[0:m_left, 6]

            zmm24 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 7*cs_a );    // zmm24 = abuf[0:m_left, 7]

            // Performing the operation y := y + A * (alpha * x) and storing the
            // result to intermediate registers.
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm10, zmm0, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm12, zmm1, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm14, zmm2, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm16, zmm3, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[3])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm18, zmm4, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 4] * (alpha*x[4])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm5, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 5] * (alpha*x[5])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm6, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 6] * (alpha*x[6])
            zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm24, zmm7, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 7] * (alpha*x[7])

            // Store the intermediate result back to the y buffer.
            _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm8 );               // ybuf[0:m_left] = zmm8
        }   // End of j-loop, m-iters.

        xbuf += 8*incx;     // Move xbuf to the next block of 8 elements.
    }   // End of i-loop, n-iters.

    // Move a to the beginning of the next block of columns.
    a = a + 8*n8_iter*cs_a;
    // Move xbuf to the beginning of the next block of elements.
    x = x + 8*n8_iter*incx;

    // Invoke respective fringe kernel based on the value of n_left.
    if ( n_left )
    {
        m_ker_fp[n_left]
        (
          conja,
          conjx,
          m,
          n_left,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          y, incy,
          cntx
        );
    }
}

/**
 * bli_dgemv_n_zen_int_16mx7_avx512(...) acts as a wrapper to invoke following
 * fringe kernels in order to handle the remaining columns when n_left = 7:
 * bli_dgemv_n_zen_int_16mx4_avx512(...)
 * bli_dgemv_n_zen_int_16mx2_avx512(...)
 * bli_dgemv_n_zen_int_16mx1_avx512(...)
 */
void bli_dgemv_n_zen_int_16mx7_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    bli_dgemv_n_zen_int_16mx4_avx512
    (
      conja,
      conjx,
      m,
      4,    // Passing n as 4 since we only want to handle 4 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );

    a = a + 4*cs;   // Move a to the beginning of the next block of columns.
    x = x + 4*incx; // Move xbuf to the beginning of the next block of elements.

    bli_dgemv_n_zen_int_16mx2_avx512
    (
      conja,
      conjx,
      m,
      2,    // Passing n as 2 since we only want to handle 2 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );

    a = a + 2*cs;   // Move a to the beginning of the next block of columns.
    x = x + 2*incx; // Move xbuf to the beginning of the next block of elements.

    bli_dgemv_n_zen_int_16mx1_avx512
    (
      conja,
      conjx,
      m,
      1,    // Passing n as 1 since we only want to handle 1 column.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );
}

/**
 * bli_dgemv_n_zen_int_16mx6_avx512(...) acts as a wrapper to invoke following
 * fringe kernels in order to handle the remaining columns when n_left = 6:
 * bli_dgemv_n_zen_int_16mx4_avx512(...)
 * bli_dgemv_n_zen_int_16mx2_avx512(...)
 */
void bli_dgemv_n_zen_int_16mx6_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    bli_dgemv_n_zen_int_16mx4_avx512
    (
      conja,
      conjx,
      m,
      4,    // Passing n as 4 since we only want to handle 4 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );

    a = a + 4*cs;   // Move a to the beginning of the next block of columns.
    x = x + 4*incx; // Move xbuf to the beginning of the next block of elements.

    bli_dgemv_n_zen_int_16mx2_avx512
    (
      conja,
      conjx,
      m,
      2,    // Passing n as 2 since we only want to handle 2 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );
}

/**
 * bli_dgemv_n_zen_int_16mx5_avx512(...) acts as a wrapper to invoke following
 * fringe kernels in order to handle the remaining columns when n_left = 5:
 * bli_dgemv_n_zen_int_16mx4_avx512(...)
 * bli_dgemv_n_zen_int_16mx1_avx512(...)
 */
void bli_dgemv_n_zen_int_16mx5_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    bli_dgemv_n_zen_int_16mx4_avx512
    (
      conja,
      conjx,
      m,
      4,    // Passing n as 4 since we only want to handle 4 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );

    a = a + 4*cs;   // Move a to the beginning of the next block of columns.
    x = x + 4*incx; // Move xbuf to the beginning of the next block of elements.

    bli_dgemv_n_zen_int_16mx1_avx512
    (
      conja,
      conjx,
      m,
      1,    // Passing n as 1 since we only want to handle 2 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );
}

/**
 * The fringe kernel bli_dgemv_n_zen_int_16mx4_avx512(...) handles DGEMV
 * operation by breaking A matrix into blocks of size 16x4, i.e., 16 rows and
 * 4 columns.
 */
void bli_dgemv_n_zen_int_16mx4_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t j;

    /**
     * m16_iter: Number of iterations while handling 16 rows at once.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows after handling blocks of 16xN and 8xN.
     */
    dim_t m16_iter = m / 16;
    dim_t m_left   = m % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // x vector
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y vector
    __m512d zmm8, zmm9;

    // A matrix
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;

    // Scale by alpha and broadcast 4 elements from x vector.
    zmm0 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );   // zmm0 = alpha*x[0*incx]
    zmm1 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );   // zmm1 = alpha*x[1*incx]
    zmm2 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );   // zmm2 = alpha*x[2*incx]
    zmm3 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );   // zmm3 = alpha*x[3*incx]

    // Loop over the m dimension, i.e., rows of A matrix and elements in y.
    // This loop handles A matrix in blocks of 16x4, calculates the
    // intermediate results and stores it back into y.
    for ( j = 0; j < m16_iter; ++j )
    {
        // Load 16 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[ 0:7]
        zmm9 = _mm512_loadu_pd( ybuf + 8 );                 // zmm9 = ybuf[8:15]

        // Load 16 rows and 4 columns from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[ 0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf + 8*rs_a + 0*cs_a );  // zmm11 = abuf[8:15, 0]

        zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[ 0:7, 1]
        zmm13 = _mm512_loadu_pd( abuf + 8*rs_a + 1*cs_a );  // zmm13 = abuf[8:15, 1]

        zmm14 = _mm512_loadu_pd( abuf + 0*rs_a + 2*cs_a );  // zmm14 = abuf[ 0:7, 2]
        zmm15 = _mm512_loadu_pd( abuf + 8*rs_a + 2*cs_a );  // zmm15 = abuf[8:15, 2]

        zmm16 = _mm512_loadu_pd( abuf + 0*rs_a + 3*cs_a );  // zmm16 = abuf[ 0:7, 3]
        zmm17 = _mm512_loadu_pd( abuf + 8*rs_a + 3*cs_a );  // zmm17 = abuf[8:15, 3]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 0] * (alpha*x[0])
        zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 1] * (alpha*x[1])
        zmm8 = _mm512_fmadd_pd( zmm14, zmm2, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 2] * (alpha*x[2])
        zmm8 = _mm512_fmadd_pd( zmm16, zmm3, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 3] * (alpha*x[3])

        zmm9 = _mm512_fmadd_pd( zmm11, zmm0, zmm9 );        // ybuf[8:15] += abuf[8:15, 0] * (alpha*x[0])
        zmm9 = _mm512_fmadd_pd( zmm13, zmm1, zmm9 );        // ybuf[8:15] += abuf[8:15, 1] * (alpha*x[1])
        zmm9 = _mm512_fmadd_pd( zmm15, zmm2, zmm9 );        // ybuf[8:15] += abuf[8:15, 2] * (alpha*x[2])
        zmm9 = _mm512_fmadd_pd( zmm17, zmm3, zmm9 );        // ybuf[8:15] += abuf[8:15, 3] * (alpha*x[3])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[ 0:7] = zmm8
        _mm512_storeu_pd( ybuf + 8*incy, zmm9 );            // ybuf[8:15] = zmm9

        ybuf += 16*incy;    // Move ybuf to the next block of 16 elements.
        abuf += 16*rs_a;    // Move abuf to the next block of 16 rows.
    }

    // If m8_iter is non-zero, handle the remaining 8 rows.
    if ( m8_iter )
    {
        // Load 8 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[0:7]

        // Load 8 rows and 4 columns from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[0:7, 0]

        zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[0:7, 1]

        zmm14 = _mm512_loadu_pd( abuf + 0*rs_a + 2*cs_a );  // zmm14 = abuf[0:7, 2]

        zmm16 = _mm512_loadu_pd( abuf + 0*rs_a + 3*cs_a );  // zmm16 = abuf[0:7, 3]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[0:7] += abuf[0:7, 0] * (alpha*x[0])
        zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[0:7] += abuf[0:7, 1] * (alpha*x[1])
        zmm8 = _mm512_fmadd_pd( zmm14, zmm2, zmm8 );        // ybuf[0:7] += abuf[0:7, 2] * (alpha*x[2])
        zmm8 = _mm512_fmadd_pd( zmm16, zmm3, zmm8 );        // ybuf[0:7] += abuf[0:7, 3] * (alpha*x[3])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[0:7] = zmm8

        ybuf += 8*incy;     // Move ybuf to the next block of 8 elements.
        abuf += 8*rs_a;     // Move ybuf to the next block of 8 elements.
    }

    // If m_left is non-zero, handle the remaining rows using masked operations.
    if ( m_left )
    {
        // Generate the mask based on the value of m_left.
        __mmask8 m_mask = (1 << m_left) - 1;

        // Load m_left elements from y.
        zmm8 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );              // zmm8 = ybuf[0:m_left]

        // Load m_left rows and 8 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 0*cs_a );    // zmm10 = abuf[0:m_left, 0]

        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 1*cs_a );    // zmm12 = abuf[0:m_left, 1]

        zmm14 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 2*cs_a );    // zmm14 = abuf[0:m_left, 2]

        zmm16 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 3*cs_a );    // zmm16 = abuf[0:m_left, 3]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm10, zmm0, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm12, zmm1, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm14, zmm2, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm16, zmm3, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[3])

        // Store the intermediate result back to the y buffer.
        _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm8 );               // ybuf[0:m_left] = zmm8
    }
}

/**
 * bli_dgemv_n_zen_int_16mx3_avx512(...) acts as a wrapper to invoke following
 * fringe kernels in order to handle the remaining columns when n_left = 3:
 * bli_dgemv_n_zen_int_16mx2_avx512(...)
 * bli_dgemv_n_zen_int_16mx1_avx512(...)
 */
void bli_dgemv_n_zen_int_16mx3_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    bli_dgemv_n_zen_int_16mx2_avx512
    (
      conja,
      conjx,
      m,
      2,    // Passing n as 2 since we only want to handle 2 columns.
      alpha,
      a, rs, cs,
      x, incx,
      beta,
      y, incy,
      cntx
    );

    a = a + 2*cs;   // Move a to the beginning of the next block of columns.
    x = x + 2*incx; // Move xbuf to the beginning of the next block of elements.

    bli_dgemv_n_zen_int_16mx1_avx512
    (
        conja,
        conjx,
        m,
        1,  // Passing n as 1 since we only want to handle 1 column.
        alpha,
        a, rs, cs,
        x, incx,
        beta,
        y, incy,
        cntx
    );
}

/**
 * The fringe kernel bli_dgemv_n_zen_int_16mx2_avx512(...) handles DGEMV
 * operation by breaking A matrix into blocks of size 16x2, i.e., 16 rows and
 * 2 columns.
 */
void bli_dgemv_n_zen_int_16mx2_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t j;

    /**
     * m16_iter: Number of iterations while handling 16 rows at once.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows after handling blocks of 16xN and 8xN.
     */
    dim_t m16_iter = m / 16;
    dim_t m_left   = m % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // x vector
    __m512d zmm0, zmm1;

    // y vector
    __m512d zmm8, zmm9;

    // A matrix
    __m512d zmm10, zmm11, zmm12, zmm13;

    // Scale by alpha and broadcast 2 elements from x vector.
    zmm0 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );   // zmm0 = alpha*x[0*incx]
    zmm1 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );   // zmm1 = alpha*x[1*incx]

    // Loop over the m dimension, i.e., rows of A matrix and elements in y.
    // This loop handles A matrix in blocks of 16x2, calculates the
    // intermediate results and stores it back into y.
    for ( j = 0; j < m16_iter; ++j )
    {
        // Load 16 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[ 0:7]
        zmm9 = _mm512_loadu_pd( ybuf + 8 );                 // zmm9 = ybuf[8:15]

        // Load 16 rows and 2 columns from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[ 0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf + 8*rs_a + 0*cs_a );  // zmm11 = abuf[8:15, 0]

        zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[ 0:7, 1]
        zmm13 = _mm512_loadu_pd( abuf + 8*rs_a + 1*cs_a );  // zmm13 = abuf[8:15, 1]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 0] * (alpha*x[0])
        zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 1] * (alpha*x[1])

        zmm9 = _mm512_fmadd_pd( zmm11, zmm0, zmm9 );        // ybuf[8:15] += abuf[8:15, 0] * (alpha*x[0])
        zmm9 = _mm512_fmadd_pd( zmm13, zmm1, zmm9 );        // ybuf[8:15] += abuf[8:15, 1] * (alpha*x[1])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[ 0:7] = zmm8
        _mm512_storeu_pd( ybuf + 8*incy, zmm9 );            // ybuf[8:15] = zmm9

        ybuf += 16*incy;    // Move ybuf to the next block of 16 elements.
        abuf += 16*rs_a;    // Move abuf to the next block of 16 rows.
    }

    // If m8_iter is non-zero, handle the remaining 8 rows.
    if ( m8_iter )
    {
        // Load 8 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[0:7]

        // Load 8 rows and 2 columns from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[0:7, 0]

        zmm12 = _mm512_loadu_pd( abuf + 0*rs_a + 1*cs_a );  // zmm12 = abuf[0:7, 1]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[0:7] += abuf[0:7, 0] * (alpha*x[0])
        zmm8 = _mm512_fmadd_pd( zmm12, zmm1, zmm8 );        // ybuf[0:7] += abuf[0:7, 1] * (alpha*x[1])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[0:7] = zmm8

        ybuf += 8*incy;     // Move ybuf to the next block of 8 elements.
        abuf += 8*rs_a;     // Move ybuf to the next block of 8 elements.
    }

    // If m_left is non-zero, handle the remaining rows using masked operations.
    if ( m_left )
    {
        // Generate the mask based on the value of m_left.
        __mmask8 m_mask = (1 << m_left) - 1;

        // Load m_left elements from y.
        zmm8 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );              // zmm8 = ybuf[0:m_left]

        // Load m_left rows and 2 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 0*cs_a );    // zmm10 = abuf[0:m_left, 0]

        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 1*cs_a );    // zmm12 = abuf[0:m_left, 1]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm10, zmm0, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm12, zmm1, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])

        // Store the intermediate result back to the y buffer.
        _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm8 );               // ybuf[0:m_left] = zmm8
    }
}

/**
 * The fringe kernel bli_dgemv_n_zen_int_16mx1_avx512(...) handles DGEMV
 * operation by breaking A matrix into blocks of size 16x1, i.e., 16 rows and
 * 1 column.
 */
void bli_dgemv_n_zen_int_16mx1_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t j;

    /**
     * m16_iter: Number of iterations while handling 16 rows at once.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows after handling blocks of 16xN and 8xN.
     */
    dim_t m16_iter = m / 16;
    dim_t m_left   = m % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // x vector
    __m512d zmm0;

    // y vector
    __m512d zmm8, zmm9;

    // A matrix
    __m512d zmm10, zmm11;

    // Scale by alpha and broadcast 1 element from x vector.
    zmm0 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );   // zmm0 = alpha*x[0*incx]

    // Loop over the m dimension, i.e., rows of A matrix and elements in y.
    // This loop handles A matrix in blocks of 16x1, calculates the
    // intermediate results and stores it back into y.
    for ( j = 0; j < m16_iter; ++j )
    {
        // Load 16 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[ 0:7]
        zmm9 = _mm512_loadu_pd( ybuf + 8 );                 // zmm9 = ybuf[8:15]

        // Load 16 rows and 1 column from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[ 0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf + 8*rs_a + 0*cs_a );  // zmm11 = abuf[8:15, 0]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[ 0:7] += abuf[ 0:7, 0] * (alpha*x[0])

        zmm9 = _mm512_fmadd_pd( zmm11, zmm0, zmm9 );        // ybuf[8:15] += abuf[8:15, 0] * (alpha*x[0])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[ 0:7] = zmm8
        _mm512_storeu_pd( ybuf + 8*incy, zmm9 );            // ybuf[8:15] = zmm9

        ybuf += 16*incy;    // Move ybuf to the next block of 16 elements.
        abuf += 16*rs_a;    // Move abuf to the next block of 16 rows.
    }

    // If m8_iter is non-zero, handle the remaining 8 rows.
    if ( m8_iter )
    {
        // Load 8 elements from y.
        zmm8 = _mm512_loadu_pd( ybuf + 0 );                 // zmm8 = ybuf[0:7]

        // Load 8 rows and 1 column from A.
        zmm10 = _mm512_loadu_pd( abuf + 0*rs_a + 0*cs_a );  // zmm10 = abuf[0:7, 0]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_fmadd_pd( zmm10, zmm0, zmm8 );        // ybuf[0:7] += abuf[0:7, 0] * (alpha*x[0])

        // Store the intermediate result back to the y buffer.
        _mm512_storeu_pd( ybuf + 0*incy, zmm8 );            // ybuf[0:7] = zmm8

        ybuf += 8*incy;     // Move ybuf to the next block of 8 elements.
        abuf += 8*rs_a;     // Move ybuf to the next block of 8 elements.
    }

    // If m_left is non-zero, handle the remaining rows using masked operations.
    if ( m_left )
    {
        // Generate the mask based on the value of m_left.
        __mmask8 m_mask = (1 << m_left) - 1;

        // Load m_left elements from y.
        zmm8 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );              // zmm8 = ybuf[0:m_left]

        // Load m_left rows and 2 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*rs_a + 0*cs_a );    // zmm10 = abuf[0:m_left, 0]

        // Performing the operation y := y + A * (alpha * x) and storing the
        // result to intermediate registers.
        zmm8 = _mm512_maskz_fmadd_pd( m_mask, zmm10, zmm0, zmm8 );          // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])

        // Store the intermediate result back to the y buffer.
        _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm8 );               // ybuf[0:m_left] = zmm8
    }
}


/**
 * ----------------------------------------------------------------------------
 * -------------------------- n-biased DGEMV kernels --------------------------
 * ----------------------------------------------------------------------------
 * - The n-biased kernels iterate over blocks of size 32x8 keeping the
 *   m-dimension(32) fixed to calculate the final result for the respective m
 *   elements of y vector.
 * - These kernels handle y scaling by beta and are optimized for cases where
 *   n < 5.
 * - The kernels follow the below naming scheme:
 *   bli_dgemv_n_zen_int_<M_BLOCK>x<N_BLOCK>n_avx512
 *   where, dgemv_n denotes NO_TRANSPOSE implementation.
 * - The main kernel, bli_dgemv_n_zen_int_32x8n_avx512(...), handles the DGEMV
 *   operation by breaking the A matrix into blocks of 32x8, i.e., 32 rows and
 *   8 columns, and traversing in the n dimension keeping m(32) fixed.
 *
 *                                        j=0    j=1    ...
 *      |----|          |----|         |------|------|-------|     |---|
 *      |    |          |    |         |      |      |       |     | 8 | j=0
 *  i=0 | 32 |      i=0 | 32 |     i=0 | 32x8 | 32x8 |       |     |---|
 *      |    |          |    |         |      |      |       |     | 8 | j=1
 *      |----|          |----|         |------|------|       |     |---|
 *      |    |  :=      |    |  +      |      |      |       |  +  |   | ...
 *  i=1 | 32 |      i=1 | 32 |     i=1 | 32x8 | 32x8 |       |     |   |
 *      |----|          |----|         |      |      |       |     |   |
 *      |    |          |    |         |------|------|       |     |   |
 *  ... |    |      ... |    |     ... |      |      |       |     |   |
 *      |    |          |    |         |      |      |       |     |   |
 *      |----|          |----|         |------|------|-------|     |---|
 *        y               y                      A                   x
 *
 * y vector:
 * y0, y1, y2, y3, ...
 *
 * A matrix:
 * a00, a01, a02, a03, ...
 * a10, a11, a12, a13, ...
 * a20, a21, a22, a23, ...
 * a30, a31, a32, a33, ...
 * ...
 *
 * x vector:
 * x0, x1, x2, x3, ...
 *
 * Outer Loop (i-loop) over m-dimension:
 *     - Initialize intermediate registers.
 *       zmm0 = 0, 0, 0, 0, ...
 *       ...
 *       zmm3 = 0, 0, 0, 0, ...
 *
 *     - Inner Loop (j-loop) over n-dimension:
 *         - Scale and broadcast 8 elements from x vector.
 *           zmm26 = alpha*x0, alpha*x0, alpha*x0, alpha*x0, ...    // zmm26 = alpha*x[0*incx]
 *           zmm27 = alpha*x1, alpha*x1, alpha*x1, alpha*x1, ...    // zmm27 = alpha*x[1*incx]
 *           ...
 *           zmm9  = alpha*x7, alpha*x7, alpha*x7, alpha*x7, ...    // zmm9  = alpha*x[7*incx]
 *
 *         - Load 32 rows & 8 columns from A matrix.
 *           zmm10 = a( 0,0), a( 1,0), a( 2,0), a( 3,0), ...        // zmm10 = abuf[  0:7, 0]
 *           zmm11 = a( 8,0), a( 9,0), a(10,0), a(11,0), ...        // zmm13 = abuf[ 8:15, 0]
 *           zmm12 = a(16,0), a(17,0), a(18,0), a(19,0), ...        // zmm13 = abuf[16:23, 0]
 *           zmm13 = a(24,0), a(25,0), a(26,0), a(27,0), ...        // zmm13 = abuf[24:31, 0]
 *           ...
 *           zmm22 = a( 0,0), a( 1,0), a( 2,0), a( 3,0), ...        // zmm22 = abuf[  0:7, 7]
 *           zmm23 = a( 8,0), a( 9,0), a(10,0), a(11,0), ...        // zmm23 = abuf[ 8:15, 7]
 *           zmm24 = a(16,0), a(17,0), a(18,0), a(19,0), ...        // zmm24 = abuf[16:23, 7]
 *           zmm25 = a(24,0), a(25,0), a(26,0), a(27,0), ...        // zmm25 = abuf[24:31, 7]
 *
 *         - Perform the operation, y := y + A * (alpha * x) and store the
 *           result in intermediate registers.
 *           zmm0 = zmm0 + zmm10 * zmm26                            // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
 *           zmm0 = zmm0 + zmm14 * zmm27                            // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
 *           ...
 *           zmm0 = zmm0 + zmm18 * zmm8                             // ybuf[  0:7] += abuf[  0:7, 6] * (alpha*x[6])
 *           zmm0 = zmm0 + zmm22 * zmm9                             // ybuf[  0:7] += abuf[  0:7, 7] * (alpha*x[7])
 *
 *           ...
 *
 *           zmm3 = zmm3 + zmm13 * zmm26                            // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])
 *           zmm3 = zmm3 + zmm17 * zmm27                            // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])
 *           ...
 *           zmm3 = zmm3 + zmm21 * zmm8                             // ybuf[24:31] += abuf[24:31, 6] * (alpha*x[6])
 *           zmm3 = zmm3 + zmm25 * zmm9                             // ybuf[24:31] += abuf[24:31, 7] * (alpha*x[7])
 *
 *         - Move abuf pointer to the next block of 8 columns.
 *           abuf += 8*cs_a;
 *
 *         - Update xbuf to point to the next block of 8 elements of x.
 *           xbuf += 8*incx;
 *
 *     - If beta = 0, store the result directly back to y buffer.
 *       Else, scale y by beta, add the intermediate result and store to y buffer.
 *       zmm30 = beta, beta, beta, beta, ...
 *
 *       zmm4 = y0, y1, y2, y3, ...
 *       ...
 *       zmm7 = y24, y25, y26, y27, ...
 *
 *       Performing y := beta*y + A * (alpha * x)
 *       zmm0 = zmm0 + zmm4 * zmm30
 *       ...
 *       zmm3 = zmm3 + zmm7 * zmm30
 *
 *     - Store the result to y buffer.
 *
 *     - Update ybuf to point to the next block of 32 elements of y.
 *       ybuf += 32*incy;
 */
void bli_dgemv_n_zen_int_32x8n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i, j;

    /**
     * m32_iter: Number of iterations while handling 32 rows at once.
     * m16_iter: Number of iterations while handling 16 rows at once. Can only
     *           be either 0 or 1.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows.
     */
    dim_t m32_iter = m / 32;
    dim_t m_left   = m % 32;
    dim_t m16_iter = m_left / 16;
          m_left   = m_left % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // If m16_iter is non-zero, calculate the result for the 16x8 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m16_iter )
    {
        bli_dgemv_n_zen_int_16x8n_avx512
        (
          conja,
          conjx,
          16,   // Passing m as 16 since we only want to handle 16 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 16*m16_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*m16_iter*incy;
    }

    // If m8_iter is non-zero, calculate the result for the 8x8 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m8_iter )
    {
        bli_dgemv_n_zen_int_8x8n_avx512
        (
          conja,
          conjx,
          8,    // Passing m as 8 since we only want to handle 8 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 8*m8_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*m8_iter*incy;
    }

    // If m_left is non-zero, calculate the result for the remaining rows.
    if ( m_left )
    {
        bli_dgemv_n_zen_int_m_leftx8n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += m_left*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += m_left*incy;
    }

    /**
     * n8_iter: Number of iterations while handling 8 columns at once.
     * n4_iter: Number of iterations while handling 4 columns at once. Can only
     *          be either 0 or 1.
     * n_left : Number of leftover columns.
     */
    dim_t n8_iter  = n / 8;
    dim_t n_left   = n % 8;
    dim_t n4_iter  = n_left / 4;
          n_left   = n_left % 4;

    // intermediate registers
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y
    __m512d zmm4, zmm5, zmm6, zmm7;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;
    __m512d zmm30, zmm31, zmm8,  zmm9;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;
    __m512d zmm18, zmm19, zmm20, zmm21;
    __m512d zmm22, zmm23, zmm24, zmm25;

    // Outer loop over the m dimension, i.e., rows of A matrix and the elements
    // in y vector.
    for ( i = 0; i < m32_iter; ++i )
    {
        // Initialize the intermediate registers to zero for every m iteration.
        zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
        zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]
        zmm2 = _mm512_setzero_pd();     // ybuf[16:23]
        zmm3 = _mm512_setzero_pd();     // ybuf[24:31]

        // Move abuf to the beginning of the next block of rows.
        abuf = a + 32*i*rs_a;
        // Initialize xbuf to the beginning of x vector for every iteration.
        xbuf = x;

        // Inner loop over the n dimension, i.e., columns of A matrix and the
        // elements in x vector.
        for ( j = 0; j < n8_iter; ++j )
        {
            // Scale by alpha and broadcast 8 elements from x vector.
            zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
            zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
            zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
            zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]
            zmm30 = _mm512_set1_pd( *alpha * (*(xbuf + 4*incx)) );  // zmm30 = alpha*x[4*incx]
            zmm31 = _mm512_set1_pd( *alpha * (*(xbuf + 5*incx)) );  // zmm31 = alpha*x[5*incx]
            zmm8  = _mm512_set1_pd( *alpha * (*(xbuf + 6*incx)) );  // zmm8  = alpha*x[6*incx]
            zmm9  = _mm512_set1_pd( *alpha * (*(xbuf + 7*incx)) );  // zmm9  = alpha*x[7*incx]

            // Load 32 rows and 8 columns from A.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
            zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
            zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
            zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

            // Load 32 rows from the next column.
            zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
            zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
            zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
            zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
            zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
            zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
            zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

            // Load 32 rows from the next column.
            zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
            zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]
            zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 2*cs_a );     // zmm20 = abuf[16:23, 2]
            zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 2*cs_a );     // zmm21 = abuf[24:31, 2]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
            zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
            zmm2 = _mm512_fmadd_pd( zmm28, zmm20, zmm2 );           // ybuf[16:23] += abuf[16:23, 2] * (alpha*x[2])
            zmm3 = _mm512_fmadd_pd( zmm28, zmm21, zmm3 );           // ybuf[24:31] += abuf[24:31, 2] * (alpha*x[2])

            // Load 32 rows from the next column.
            zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
            zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]
            zmm24 = _mm512_loadu_pd( abuf + 16*rs_a + 3*cs_a );     // zmm24 = abuf[16:23, 3]
            zmm25 = _mm512_loadu_pd( abuf + 24*rs_a + 3*cs_a );     // zmm25 = abuf[24:31, 3]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
            zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])
            zmm2 = _mm512_fmadd_pd( zmm29, zmm24, zmm2 );           // ybuf[16:23] += abuf[16:23, 3] * (alpha*x[3])
            zmm3 = _mm512_fmadd_pd( zmm29, zmm25, zmm3 );           // ybuf[24:31] += abuf[24:31, 3] * (alpha*x[3])

            // Load 32 rows from the next column.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 4*cs_a );     // zmm10 = abuf[  0:7, 4]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 4*cs_a );     // zmm11 = abuf[ 8:15, 4]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 4*cs_a );     // zmm12 = abuf[16:23, 4]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 4*cs_a );     // zmm13 = abuf[24:31, 4]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm30, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 4] * (alpha*x[4])
            zmm1 = _mm512_fmadd_pd( zmm30, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 4] * (alpha*x[4])
            zmm2 = _mm512_fmadd_pd( zmm30, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 4] * (alpha*x[4])
            zmm3 = _mm512_fmadd_pd( zmm30, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 4] * (alpha*x[4])

            // Load 32 rows from the next column.
            zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 5*cs_a );     // zmm14 = abuf[  0:7, 5]
            zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 5*cs_a );     // zmm15 = abuf[ 8:15, 5]
            zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 5*cs_a );     // zmm16 = abuf[16:23, 5]
            zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 5*cs_a );     // zmm17 = abuf[24:31, 5]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm31, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 5] * (alpha*x[5])
            zmm1 = _mm512_fmadd_pd( zmm31, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 5] * (alpha*x[5])
            zmm2 = _mm512_fmadd_pd( zmm31, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 5] * (alpha*x[5])
            zmm3 = _mm512_fmadd_pd( zmm31, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 5] * (alpha*x[5])

            // Load 32 rows from the next column.
            zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 6*cs_a );     // zmm18 = abuf[  0:7, 6]
            zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 6*cs_a );     // zmm19 = abuf[ 8:15, 6]
            zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 6*cs_a );     // zmm20 = abuf[16:23, 6]
            zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 6*cs_a );     // zmm21 = abuf[24:31, 6]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm8, zmm18, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 6] * (alpha*x[6])
            zmm1 = _mm512_fmadd_pd( zmm8, zmm19, zmm1 );            // ybuf[ 8:15] += abuf[ 8:15, 6] * (alpha*x[6])
            zmm2 = _mm512_fmadd_pd( zmm8, zmm20, zmm2 );            // ybuf[16:23] += abuf[16:23, 6] * (alpha*x[6])
            zmm3 = _mm512_fmadd_pd( zmm8, zmm21, zmm3 );            // ybuf[24:31] += abuf[24:31, 6] * (alpha*x[6])

            // Load 32 rows from the next column.
            zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 7*cs_a );     // zmm22 = abuf[  0:7, 7]
            zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 7*cs_a );     // zmm23 = abuf[ 8:15, 7]
            zmm24 = _mm512_loadu_pd( abuf + 16*rs_a + 7*cs_a );     // zmm24 = abuf[16:23, 7]
            zmm25 = _mm512_loadu_pd( abuf + 24*rs_a + 7*cs_a );     // zmm25 = abuf[24:31, 7]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm9, zmm22, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 7] * (alpha*x[7])
            zmm1 = _mm512_fmadd_pd( zmm9, zmm23, zmm1 );            // ybuf[ 8:15] += abuf[ 8:15, 7] * (alpha*x[7])
            zmm2 = _mm512_fmadd_pd( zmm9, zmm24, zmm2 );            // ybuf[16:23] += abuf[16:23, 7] * (alpha*x[7])
            zmm3 = _mm512_fmadd_pd( zmm9, zmm25, zmm3 );            // ybuf[24:31] += abuf[24:31, 7] * (alpha*x[7])

            abuf += 8*cs_a;     // Move abuf to the next block of 8 columns.
            xbuf += 8*incx;     // Move xbuf to the next block of 8 elements.
        }

        // If n4_iter is non-zero, handle the remaining 4 columns.
        if ( n4_iter )      // Performing GEMV in blocks of 32x4.
        {
            // Scale by alpha and broadcast 4 elements from x vector.
            zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
            zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
            zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
            zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]

            // Load 32 rows and 4 columns from A.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
            zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
            zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
            zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

            // Load 32 rows from the next column.
            zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
            zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
            zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
            zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
            zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
            zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
            zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

            // Load 32 rows from the next column.
            zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
            zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]
            zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 2*cs_a );     // zmm20 = abuf[16:23, 2]
            zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 2*cs_a );     // zmm21 = abuf[24:31, 2]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
            zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
            zmm2 = _mm512_fmadd_pd( zmm28, zmm20, zmm2 );           // ybuf[16:23] += abuf[16:23, 2] * (alpha*x[2])
            zmm3 = _mm512_fmadd_pd( zmm28, zmm21, zmm3 );           // ybuf[24:31] += abuf[24:31, 2] * (alpha*x[2])

            // Load 32 rows from the next column.
            zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
            zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]
            zmm24 = _mm512_loadu_pd( abuf + 16*rs_a + 3*cs_a );     // zmm24 = abuf[16:23, 3]
            zmm25 = _mm512_loadu_pd( abuf + 24*rs_a + 3*cs_a );     // zmm25 = abuf[24:31, 3]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
            zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])
            zmm2 = _mm512_fmadd_pd( zmm29, zmm24, zmm2 );           // ybuf[16:23] += abuf[16:23, 3] * (alpha*x[3])
            zmm3 = _mm512_fmadd_pd( zmm29, zmm25, zmm3 );           // ybuf[24:31] += abuf[24:31, 3] * (alpha*x[3])

            abuf += 4*cs_a;     // Move abuf to the next block of 4 columns.
            xbuf += 4*incx;     // Move xbuf to the next block of 4 elements.
        }

        // If n_left is non-zero, handle the remaining columns.
        if ( n_left == 3 )      // Performing GEMV in blocks of 32x3.
        {
            // Scale by alpha and broadcast 3 elements from x vector.
            zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
            zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
            zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

            // Load 32 rows and 3 columns from A.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
            zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
            zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
            zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

            // Load 32 rows from the next column.
            zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
            zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
            zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
            zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
            zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
            zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
            zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

            // Load 32 rows from the next column.
            zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
            zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]
            zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 2*cs_a );     // zmm20 = abuf[16:23, 2]
            zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 2*cs_a );     // zmm21 = abuf[24:31, 2]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
            zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
            zmm2 = _mm512_fmadd_pd( zmm28, zmm20, zmm2 );           // ybuf[16:23] += abuf[16:23, 2] * (alpha*x[2])
            zmm3 = _mm512_fmadd_pd( zmm28, zmm21, zmm3 );           // ybuf[24:31] += abuf[24:31, 2] * (alpha*x[2])
        }
        else if ( n_left == 2 )     // Performing GEMV in blocks of 32x2.
        {
            // Scale by alpha and broadcast 2 elements from x vector.
            zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
            zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

            // Load 32 rows and 2 columns from A.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
            zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
            zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
            zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

            // Load 32 rows from the next column.
            zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
            zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
            zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
            zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
            zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
            zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
            zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])
        }
        else if ( n_left == 1 )     // Performing GEMV in blocks of 32x1.
        {
            // Scale by alpha and broadcast 1 element from x vector.
            zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

            // Load 16 rows and 1 columns from A.
            zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
            zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
            zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
            zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

            // Performing the operation intermediate_register += A * (alpha * x)
            // and storing the result back to the intermediate registers.
            zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
            zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
            zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
            zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])
        }

        /**
         * If beta = 0, then y vector should not be read, only set.
         * Else, load y, scale by beta, add the intermediate results and store
         * the final result back to y buffer.
         */
        if ( !bli_deq0( *beta ) )
        {
            // Broadcast beta to a register.
            zmm30 = _mm512_set1_pd( *beta );                        // zmm30 = beta, beta, beta, beta, ...

            // Load 32 elements from y vector.
            zmm4 = _mm512_loadu_pd( ybuf +  0*incy );               // zmm4 = ybuf[  0:7]
            zmm5 = _mm512_loadu_pd( ybuf +  8*incy );               // zmm5 = ybuf[ 8:15]
            zmm6 = _mm512_loadu_pd( ybuf + 16*incy );               // zmm6 = ybuf[16:23]
            zmm7 = _mm512_loadu_pd( ybuf + 24*incy );               // zmm7 = ybuf[24:31]

            // Performing the operation y := beta*y + A * (alpha * x).
            zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );            // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
            zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );            // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
            zmm2 = _mm512_fmadd_pd( zmm30, zmm6, zmm2 );            // ybuf[16:23] = beta*ybuf[16:23] + (abuf[16:23, 0] * (alpha*x[0]))
            zmm3 = _mm512_fmadd_pd( zmm30, zmm7, zmm3 );            // ybuf[24:31] = beta*ybuf[24:31] + (abuf[24:31, 0] * (alpha*x[0]))
        }

        // Store the final result to y buffer.
        _mm512_storeu_pd( ybuf +  0*incy, zmm0 );                   // ybuf[  0:7] = zmm0
        _mm512_storeu_pd( ybuf +  8*incy, zmm1 );                   // ybuf[ 8:15] = zmm1
        _mm512_storeu_pd( ybuf + 16*incy, zmm2 );                   // ybuf[16:23] = zmm2
        _mm512_storeu_pd( ybuf + 24*incy, zmm3 );                   // ybuf[24:31] = zmm3

        ybuf += 32*incy;    // Move ybuf to the next block of 32 elements.
    }
}

/**
 * The bli_dgemv_n_zen_int_16x8n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 16x8, i.e., 16 rows and 8 columns,
 * and traversing in the n dimension keeping m(16) fixed.
 */
void bli_dgemv_n_zen_int_16x8n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t j;

    /**
     * n8_iter: Number of iterations while handling 8 columns at once.
     * n4_iter: Number of iterations while handling 4 columns at once. Can only
     *          be either 0 or 1.
     * n_left : Number of leftover columns.
     */
    dim_t n8_iter = n / 8;
    dim_t n_left  = n % 8;
    dim_t n4_iter = n_left / 4;
          n_left  = n_left % 4;

    // This kernel will handle sizes where m = [16, 32).
    dim_t m_left   = m % 16;

    // intermediate registers
    __m512d zmm0, zmm1;

    // y
    __m512d zmm4, zmm5;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;
    __m512d zmm30, zmm31, zmm8,  zmm9;

    // A
    __m512d zmm10, zmm11;
    __m512d zmm14, zmm15;
    __m512d zmm18, zmm19;
    __m512d zmm22, zmm23;

    // beta
    // Broadcast beta to a register.
    __m512d zmm3 = _mm512_set1_pd( *beta );     // zmm3 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
    zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]

    // Loop over the n dimension, i.e., columns of A matrix and the
    // elements in x vector.
    for ( j = 0; j < n8_iter; ++j )
    {
        // Scale by alpha and broadcast 8 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
        zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]
        zmm30 = _mm512_set1_pd( *alpha * (*(xbuf + 4*incx)) );  // zmm30 = alpha*x[4*incx]
        zmm31 = _mm512_set1_pd( *alpha * (*(xbuf + 5*incx)) );  // zmm31 = alpha*x[5*incx]
        zmm8  = _mm512_set1_pd( *alpha * (*(xbuf + 6*incx)) );  // zmm8  = alpha*x[6*incx]
        zmm9  = _mm512_set1_pd( *alpha * (*(xbuf + 7*incx)) );  // zmm9  = alpha*x[7*incx]

        // Load 16 rows and 8 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
        zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
        zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]

        zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
        zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])

        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 4*cs_a );     // zmm10 = abuf[  0:7, 4]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 4*cs_a );     // zmm11 = abuf[ 8:15, 4]

        zmm0 = _mm512_fmadd_pd( zmm30, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 4] * (alpha*x[4])
        zmm1 = _mm512_fmadd_pd( zmm30, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 4] * (alpha*x[4])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 5*cs_a );     // zmm14 = abuf[  0:7, 5]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 5*cs_a );     // zmm15 = abuf[ 8:15, 5]

        zmm0 = _mm512_fmadd_pd( zmm31, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 5] * (alpha*x[5])
        zmm1 = _mm512_fmadd_pd( zmm31, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 5] * (alpha*x[5])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 6*cs_a );     // zmm18 = abuf[  0:7, 6]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 6*cs_a );     // zmm19 = abuf[ 8:15, 6]

        zmm0 = _mm512_fmadd_pd( zmm8, zmm18, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 6] * (alpha*x[6])
        zmm1 = _mm512_fmadd_pd( zmm8, zmm19, zmm1 );            // ybuf[ 8:15] += abuf[ 8:15, 6] * (alpha*x[6])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 7*cs_a );     // zmm22 = abuf[  0:7, 7]
        zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 7*cs_a );     // zmm23 = abuf[ 8:15, 7]

        zmm0 = _mm512_fmadd_pd( zmm9, zmm22, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 7] * (alpha*x[7])
        zmm1 = _mm512_fmadd_pd( zmm9, zmm23, zmm1 );            // ybuf[ 8:15] += abuf[ 8:15, 7] * (alpha*x[7])

        abuf += 8*cs_a;     // Move abuf to the next block of 8 columns.
        xbuf += 8*incx;     // Move xbuf to the next block of 8 elements.
    }

    // If n4_iter is non-zero, handle the remaining 4 columns.
    if ( n4_iter )      // Performing GEMV in blocks of 16x4.
    {
        // Scale by alpha and broadcast 4 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
        zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]

        // Load 16 rows and 4 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
        zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
        zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]

        zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
        zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])

        abuf += 4*cs_a;     // Move abuf to the next block of 4 columns.
        xbuf += 4*incx;     // Move xbuf to the next block of 4 elements.
    }

    // If n_left is non-zero, handle the remaining columns.
    if ( n_left == 3 )      // Performing GEMV in blocks of 16x3.
    {
        // Scale by alpha and broadcast 3 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

        // Load 16 rows and 3 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
        zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
    }
    else if ( n_left == 2 )     // Performing GEMV in blocks of 16x2.
    {
        // Scale by alpha and broadcast 2 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

        // Load 16 rows and 2 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
    }
    else if ( n_left == 1 )     // Performing GEMV in blocks of 16x1.
    {
        // Scale by alpha and broadcast 1 element from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

        // Load 16 rows and 1 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
    }

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 16 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );               // zmm4 = ybuf[  0:7]
        zmm5 = _mm512_loadu_pd( ybuf +  8*incy );               // zmm5 = ybuf[ 8:15]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm3 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm3, zmm4, zmm0 );             // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
        zmm1 = _mm512_fmadd_pd( zmm3, zmm5, zmm1 );             // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );                   // ybuf[  0:7] = zmm0
    _mm512_storeu_pd( ybuf +  8*incy, zmm1 );                   // ybuf[ 8:15] = zmm1

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 16*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*incy;

        dim_t m8_iter = m_left / 8;
              m_left  = m_left % 8;

        if ( m8_iter )
        {
            bli_dgemv_n_zen_int_8x8n_avx512
            (
              conja,
              conjx,
              8,    // Passing m as 8 since we only want to handle 8 rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );

            // Move a to the beginning of the next block of rows.
            a    += 8*rs_a;
            // Move ybuf to the beginning of the next block of elements.
            ybuf += 8*incy;
        }

        // If m_left is non-zero, calculate the result for the remaining rows.
        if ( m_left )
        {
            bli_dgemv_n_zen_int_m_leftx8n_avx512
            (
              conja,
              conjx,
              m_left,   // Passing m as m_left since we only want to handle m_left rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );
        }
    }
}

/**
 * The bli_dgemv_n_zen_int_8x8n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 8x8, i.e., 8 rows and 8 columns,
 * and traversing in the n dimension keeping m(8) fixed.
 */
void bli_dgemv_n_zen_int_8x8n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t j;

    /**
     * n8_iter: Number of iterations while handling 8 columns at once.
     * n4_iter: Number of iterations while handling 4 columns at once. Can only
     *          be either 0 or 1.
     * n_left : Number of leftover columns.
     */
    dim_t n8_iter = n / 8;
    dim_t n_left  = n % 8;
    dim_t n4_iter = n_left / 4;
          n_left  = n_left % 4;

    // This kernel will handle sizes where m = [8, 16).
    dim_t m_left = m % 8;

    // intermediate registers
    __m512d zmm0;

    // y
    __m512d zmm4;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;
    __m512d zmm30, zmm31, zmm8,  zmm9;

    // A
    __m512d zmm10;
    __m512d zmm14;
    __m512d zmm18;
    __m512d zmm22;

    // beta
    // Broadcast beta to a register.
    __m512d zmm3 = _mm512_set1_pd( *beta );     // zmm3 = beta, beta, beta, beta, ...

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                 // ybuf[  0:7]

    // Loop over the n dimension, i.e., columns of A matrix and the
    // elements in x vector.
    for ( j = 0; j < n8_iter; ++j )
    {
        // Scale by alpha and broadcast 8 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
        zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]
        zmm30 = _mm512_set1_pd( *alpha * (*(xbuf + 4*incx)) );  // zmm30 = alpha*x[4*incx]
        zmm31 = _mm512_set1_pd( *alpha * (*(xbuf + 5*incx)) );  // zmm31 = alpha*x[5*incx]
        zmm8  = _mm512_set1_pd( *alpha * (*(xbuf + 6*incx)) );  // zmm8  = alpha*x[6*incx]
        zmm9  = _mm512_set1_pd( *alpha * (*(xbuf + 7*incx)) );  // zmm9  = alpha*x[7*incx]

        // Load 8 rows and 8 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]

        zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])

        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 4*cs_a );     // zmm10 = abuf[  0:7, 4]

        zmm0 = _mm512_fmadd_pd( zmm30, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 4] * (alpha*x[4])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 5*cs_a );     // zmm14 = abuf[  0:7, 5]

        zmm0 = _mm512_fmadd_pd( zmm31, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 5] * (alpha*x[5])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 6*cs_a );     // zmm18 = abuf[  0:7, 6]

        zmm0 = _mm512_fmadd_pd( zmm8, zmm18, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 6] * (alpha*x[6])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 7*cs_a );     // zmm22 = abuf[  0:7, 7]

        zmm0 = _mm512_fmadd_pd( zmm9, zmm22, zmm0 );            // ybuf[  0:7] += abuf[  0:7, 7] * (alpha*x[7])

        abuf += 8*cs_a;     // Move abuf to the next block of 8 columns.
        xbuf += 8*incx;     // Move xbuf to the next block of 8 elements.
    }

    // If n4_iter is non-zero, handle the remaining 4 columns.
    if ( n4_iter )      // Performing GEMV in blocks of 8x4.
    {
        // Scale by alpha and broadcast 4 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );
        zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );

        // Load 4 rows and 4 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]

        zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])

        abuf += 4*cs_a;     // Move abuf to the next block of 4 columns.
        xbuf += 4*incx;     // Move xbuf to the next block of 4 elements.
    }

    // If n_left is non-zero, handle the remaining columns.
    if ( n_left == 3 )      // Performing GEMV in blocks of 8x3.
    {
        // Scale by alpha and broadcast 3 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

        // Load 8 rows and 3 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
    }
    else if ( n_left == 2 )      // Performing GEMV in blocks of 8x2.
    {
        // Scale by alpha and broadcast 2 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

        // Load 8 rows and 2 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
    }
    else if ( n_left == 1 )     // Performing GEMV in blocks of 8x1.
    {
        // Scale by alpha and broadcast 1 element from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

        // Load 8 rows and 1 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
    }

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 8 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );               // zmm4 = ybuf[  0:7]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm3 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm3, zmm4, zmm0 );             // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );                   // ybuf[  0:7] = zmm0

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 8*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*incy;

        bli_dgemv_n_zen_int_m_leftx8n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );
    }
}

/**
 * The bli_dgemv_n_zen_int_m_leftx8n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of m_leftx8, i.e., m_left rows and 8 columns,
 * and traversing in the n dimension keeping m(m_left) fixed.
 */
void bli_dgemv_n_zen_int_m_leftx8n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m_left,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t cs_a = cs;

    /**
     * n8_iter: Number of iterations while handling 8 columns at once.
     * n4_iter: Number of iterations while handling 4 columns at once. Can only
     *          be either 0 or 1.
     * n_left : Number of leftover columns.
     */
    dim_t n8_iter = n / 8;
    dim_t n_left  = n % 8;
    dim_t n4_iter = n_left / 4;
          n_left  = n_left % 4;

    dim_t j;

    // intermediate registers
    __m512d zmm0;

    // x
    __m512d zmm20, zmm21, zmm22, zmm23;
    __m512d zmm30, zmm31, zmm8,  zmm9;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;

    // y
    __m512d zmm4;

    // beta
    __m512d zmm3;

    // Generate the mask based on the value of m_left.
    __mmask8 m_mask = (1 << m_left) - 1;

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();     // ybuf[0:m_left]

    // Loop over the n dimension, i.e., columns of A matrix and the
    // elements in x vector.
    for ( j = 0; j < n8_iter; ++j )
    {
        // Scale by alpha and broadcast 8 elements from x vector.
        zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );      // zmm20 = alpha*x[0*incx]
        zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );      // zmm21 = alpha*x[1*incx]
        zmm22 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );      // zmm22 = alpha*x[2*incx]
        zmm23 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );      // zmm23 = alpha*x[3*incx]
        zmm30 = _mm512_set1_pd( *alpha * (*(xbuf + 4*incx)) );      // zmm30 = alpha*x[4*incx]
        zmm31 = _mm512_set1_pd( *alpha * (*(xbuf + 5*incx)) );      // zmm31 = alpha*x[5*incx]
        zmm8  = _mm512_set1_pd( *alpha * (*(xbuf + 6*incx)) );      // zmm8  = alpha*x[6*incx]
        zmm9  = _mm512_set1_pd( *alpha * (*(xbuf + 7*incx)) );      // zmm9  = alpha*x[7*incx]

        // Load masked rows and 8 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]
        zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );     // zmm11 = abuf[0:m_left, 1]
        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 2*cs_a );     // zmm12 = abuf[0:m_left, 2]
        zmm13 = _mm512_maskz_loadu_pd( m_mask, abuf + 3*cs_a );     // zmm13 = abuf[0:m_left, 3]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm12, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm23, zmm13, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[3])

        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 4*cs_a );     // zmm10 = abuf[0:m_left, 4]
        zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 5*cs_a );     // zmm11 = abuf[0:m_left, 5]
        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 6*cs_a );     // zmm12 = abuf[0:m_left, 6]
        zmm13 = _mm512_maskz_loadu_pd( m_mask, abuf + 7*cs_a );     // zmm13 = abuf[0:m_left, 7]

        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm30, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[4])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm31, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[5])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm8,  zmm12, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[6])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm9,  zmm13, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[7])

        xbuf += 8*incx;     // Move xbuf to the next block of 8 columns.
        abuf += 8*cs_a;     // Move abuf to the next block of 8 elements.
    }

    // If n4_iter is non-zero, handle the remaining 4 columns.
    if ( n4_iter )      // Performing GEMV in blocks of m_leftx4.
    {
        // Scale by alpha and broadcast 4 elements from x vector.
        zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]
        zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );     // zmm21 = alpha*x[1*incx]
        zmm22 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx) ) );     // zmm22 = alpha*x[2*incx]
        zmm23 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx) ) );     // zmm23 = alpha*x[3*incx]

        // Load masked rows and 4 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]
        zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );     // zmm11 = abuf[0:m_left, 1]
        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 2*cs_a );     // zmm12 = abuf[0:m_left, 2]
        zmm13 = _mm512_maskz_loadu_pd( m_mask, abuf + 3*cs_a );     // zmm13 = abuf[0:m_left, 3]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm12, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm23, zmm13, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[3])

        xbuf += 4*incx;     // Move xbuf to the next block of 4 columns.
        abuf += 4*cs_a;     // Move abuf to the next block of 4 elements.
    }

    // If n_left is non-zero, handle the remaining columns.
    if ( n_left == 3 )      // Performing GEMV in blocks of m_leftx3.
    {
        // Scale by alpha and broadcast 3 elements from x vector.
        zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]
        zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );     // zmm21 = alpha*x[1*incx]
        zmm22 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx) ) );     // zmm22 = alpha*x[2*incx]

        // Load masked rows and 3 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]
        zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );     // zmm11 = abuf[0:m_left, 1]
        zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 2*cs_a );     // zmm12 = abuf[0:m_left, 2]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm12, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
    }
    else if ( n_left == 2 )     // Performing GEMV in blocks of m_leftx2.
    {
        // Scale by alpha and broadcast 2 elements from x vector.
        zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]
        zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );     // zmm21 = alpha*x[1*incx]

        // Load masked rows and 2 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]
        zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );     // zmm11 = abuf[0:m_left, 1]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
    }
    else if ( n_left == 1 )     // Performing GEMV in blocks of m_leftx1.
    {
        // Scale by alpha and broadcast 1 elements from x vector.
        zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]

        // Load masked rows and 1 columns from A.
        zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
    }

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Broadcast beta to a register.
        zmm3 = _mm512_set1_pd( *beta );                             // zmm3 = beta, beta, beta, beta, ...

        // Load m_left elements from y vector.
        zmm4 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );      // zmm4 = ybuf[0:m_left]

        // Performing the operation y := beta*y + A * (alpha * x).
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm3, zmm4, zmm0 );   // ybuf[0:m_left] = beta*ybuf[0:m_left] + (abuf[0:m_left, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm0 );           // ybuf[0:m_left] = zmm0
}

/**
 * The bli_dgemv_n_zen_int_32x4n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 32x4, i.e., 32 rows and 4 columns,
 * and traversing in the n dimension keeping m(32) fixed.
 */
void bli_dgemv_n_zen_int_32x4n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i;

    /**
     * m32_iter: Number of iterations while handling 32 rows at once.
     * m16_iter: Number of iterations while handling 16 rows at once.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows.
     */
    dim_t m32_iter = m / 32;
    dim_t m_left   = m % 32;
    dim_t m16_iter = m_left / 16;
          m_left   = m_left % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // If m16_iter is non-zero, calculate the result for the 16x4 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m16_iter )
    {
        bli_dgemv_n_zen_int_16x4n_avx512
        (
          conja,
          conjx,
          16,   // Passing m as 16 since we only want to handle 16 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 16*m16_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*m16_iter*incy;
    }

    // If m8_iter is non-zero, calculate the result for the 8x4 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m8_iter )
    {
        bli_dgemv_n_zen_int_8x4n_avx512
        (
          conja,
          conjx,
          8,    // Passing m as 8 since we only want to handle 8 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 8*m8_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*m8_iter*incy;
    }

    // If m_left is non-zero, calculate the result for the remaining rows.
    if ( m_left )
    {
        bli_dgemv_n_zen_int_m_leftx4n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += m_left*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += m_left*incy;
    }

    // intermediate registers
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y
    __m512d zmm4, zmm5, zmm6, zmm7;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;
    __m512d zmm18, zmm19, zmm20, zmm21;
    __m512d zmm22, zmm23, zmm24, zmm25;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Loop over the m dimension, i.e., rows of A matrix and the elements
    // in y vector.
    for ( i = 0; i < m32_iter; ++i )
    {
        // Initialize the intermediate registers to zero for every m iteration.
        zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
        zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]
        zmm2 = _mm512_setzero_pd();     // ybuf[16:23]
        zmm3 = _mm512_setzero_pd();     // ybuf[24:31]

        // Move abuf to the beginning of the next block of rows.
        abuf = a + 32*i*rs_a;
        // Initialize xbuf to the beginning of x vector for every iteration.
        xbuf = x;

        // Scale by alpha and broadcast 4 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
        zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]

        // Load 32 rows and 4 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
        zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
        zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
        zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
        zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
        zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
        zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
        zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
        zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]
        zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 2*cs_a );     // zmm20 = abuf[16:23, 2]
        zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 2*cs_a );     // zmm21 = abuf[24:31, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
        zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
        zmm2 = _mm512_fmadd_pd( zmm28, zmm20, zmm2 );           // ybuf[16:23] += abuf[16:23, 2] * (alpha*x[2])
        zmm3 = _mm512_fmadd_pd( zmm28, zmm21, zmm3 );           // ybuf[24:31] += abuf[24:31, 2] * (alpha*x[2])

        zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
        zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]
        zmm24 = _mm512_loadu_pd( abuf + 16*rs_a + 3*cs_a );     // zmm24 = abuf[16:23, 3]
        zmm25 = _mm512_loadu_pd( abuf + 24*rs_a + 3*cs_a );     // zmm25 = abuf[24:31, 3]

        zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
        zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])
        zmm2 = _mm512_fmadd_pd( zmm29, zmm24, zmm2 );           // ybuf[16:23] += abuf[16:23, 3] * (alpha*x[3])
        zmm3 = _mm512_fmadd_pd( zmm29, zmm25, zmm3 );           // ybuf[24:31] += abuf[24:31, 3] * (alpha*x[3])

        /**
         * If beta = 0, then y vector should not be read, only set.
         * Else, load y, scale by beta, add the intermediate results and store
         * the final result back to y buffer.
         */
        if ( !bli_deq0( *beta ) )
        {
            // Load 32 elements from y vector.
            zmm4 = _mm512_loadu_pd( ybuf +  0*incy );       // zmm4 = ybuf[  0:7]
            zmm5 = _mm512_loadu_pd( ybuf +  8*incy );       // zmm5 = ybuf[ 8:15]
            zmm6 = _mm512_loadu_pd( ybuf + 16*incy );       // zmm6 = ybuf[16:23]
            zmm7 = _mm512_loadu_pd( ybuf + 24*incy );       // zmm7 = ybuf[24:31]

            // Performing the operation y := beta*y + A * (alpha * x).
            // zmm30 = _mm512_set1_pd( *beta );
            zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );    // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
            zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );    // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
            zmm2 = _mm512_fmadd_pd( zmm30, zmm6, zmm2 );    // ybuf[16:23] = beta*ybuf[16:23] + (abuf[16:23, 0] * (alpha*x[0]))
            zmm3 = _mm512_fmadd_pd( zmm30, zmm7, zmm3 );    // ybuf[24:31] = beta*ybuf[24:31] + (abuf[24:31, 0] * (alpha*x[0]))
        }

        // Store the final result to y buffer.
        _mm512_storeu_pd( ybuf +  0*incy, zmm0 );           // ybuf[  0:7] = zmm0
        _mm512_storeu_pd( ybuf +  8*incy, zmm1 );           // ybuf[ 8:15] = zmm1
        _mm512_storeu_pd( ybuf + 16*incy, zmm2 );           // ybuf[16:23] = zmm2
        _mm512_storeu_pd( ybuf + 24*incy, zmm3 );           // ybuf[24:31] = zmm3

        ybuf += 32*incy;    // Move ybuf to the next block of 32 elements.
    }
}

/**
 * The bli_dgemv_n_zen_int_16x4n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 16x4, i.e., 16 rows and 4 columns,
 * and traversing in the n dimension keeping m(16) fixed.
 */
void bli_dgemv_n_zen_int_16x4n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [16, 32).
    dim_t m_left = m % 16;

    // intermediate registers
    __m512d zmm0, zmm1;

    // y
    __m512d zmm4, zmm5;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;

    // A
    __m512d zmm10, zmm11;
    __m512d zmm14, zmm15;
    __m512d zmm18, zmm19;
    __m512d zmm22, zmm23;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
    zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]

    // Scale by alpha and broadcast 4 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
    zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
    zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]

    // Load 16 rows and 4 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
    zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
    zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
    zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
    zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

    zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
    zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]

    zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
    zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])

    zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]
    zmm23 = _mm512_loadu_pd( abuf +  8*rs_a + 3*cs_a );     // zmm23 = abuf[ 8:15, 3]

    zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])
    zmm1 = _mm512_fmadd_pd( zmm29, zmm23, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 3] * (alpha*x[3])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 16 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );               // zmm4 = ybuf[  0:7]
        zmm5 = _mm512_loadu_pd( ybuf +  8*incy );               // zmm5 = ybuf[ 8:15]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );            // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
        zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );            // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );                   // ybuf[  0:7] = zmm0
    _mm512_storeu_pd( ybuf +  8*incy, zmm1 );                   // ybuf[ 8:15] = zmm1

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 16*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*incy;

        dim_t m8_iter = m_left / 8;
              m_left  = m_left % 8;

        if ( m8_iter )
        {
            bli_dgemv_n_zen_int_8x4n_avx512
            (
              conja,
              conjx,
              8,    // Passing m as 8 since we only want to handle 8 rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );

            // Move a to the beginning of the next block of rows.
            a    += 8*rs_a;
            // Move ybuf to the beginning of the next block of elements.
            ybuf += 8*incy;
        }

        // If m_left is non-zero, calculate the result for the remaining rows.
        if ( m_left )
        {
            bli_dgemv_n_zen_int_m_leftx4n_avx512
            (
              conja,
              conjx,
              m_left,   // Passing m as m_left since we only want to handle m_left rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );
        }
    }
}

/**
 * The bli_dgemv_n_zen_int_8x4n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 8x4, i.e., 8 rows and 4 columns,
 * and traversing in the n dimension keeping m(8) fixed.
 */
void bli_dgemv_n_zen_int_8x4n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [8, 16).
    dim_t m_left = m % 8;

    // intermediate registers
    __m512d zmm0;

    // y
    __m512d zmm4;

    // x
    __m512d zmm26, zmm27, zmm28, zmm29;

    // A
    __m512d zmm10;
    __m512d zmm14;
    __m512d zmm18;
    __m512d zmm22;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]

    // Scale by alpha and broadcast 4 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
    zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]
    zmm29 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx)) );  // zmm29 = alpha*x[3*incx]

    // Load 8 rows and 4 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

    zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]

    zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])

    zmm22 = _mm512_loadu_pd( abuf +  0*rs_a + 3*cs_a );     // zmm22 = abuf[  0:7, 3]

    zmm0 = _mm512_fmadd_pd( zmm29, zmm22, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 3] * (alpha*x[3])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 8 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );               // zmm4 = ybuf[  0:7]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );            // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );                   // ybuf[  0:7] = zmm0

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 8*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*incy;

        bli_dgemv_n_zen_int_m_leftx4n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );
    }
}

/**
 * The bli_dgemv_n_zen_int_m_leftx4n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of m_leftx4, i.e., m_left rows and 4 columns,
 * and traversing in the n dimension keeping m(m_left) fixed.
 */
void bli_dgemv_n_zen_int_m_leftx4n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m_left,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t cs_a = cs;

    // intermediate registers
    __m512d zmm0;

    // x
    __m512d zmm20, zmm21, zmm22, zmm23;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;

    // y
    __m512d zmm30;

    // beta
    __m512d zmm8;

    // Generate the mask based on the value of m_left.
    __mmask8 m_mask = (1 << m_left) - 1;

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();     // ybuf[0:m_left]

    // Scale by alpha and broadcast 4 elements from x vector.
    zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]
    zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );     // zmm21 = alpha*x[1*incx]
    zmm22 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx) ) );     // zmm22 = alpha*x[2*incx]
    zmm23 = _mm512_set1_pd( *alpha * (*(xbuf + 3*incx) ) );     // zmm23 = alpha*x[3*incx]

    // Load masked rows and 4 columns from A.
    zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]
    zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );     // zmm11 = abuf[0:m_left, 1]
    zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 2*cs_a );     // zmm12 = abuf[0:m_left, 2]
    zmm13 = _mm512_maskz_loadu_pd( m_mask, abuf + 3*cs_a );     // zmm13 = abuf[0:m_left, 3]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm12, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm23, zmm13, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 3] * (alpha*x[3])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Broadcast beta to a register.
        zmm8 = _mm512_set1_pd( *beta );                             // zmm8 = beta, beta, beta, beta, ...

        // Load m_left elements from y vector.
        zmm30 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );     // zmm4 = ybuf[0:m_left]

        // Performing the operation y := beta*y + A * (alpha * x).
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm8, zmm30, zmm0 );  // ybuf[0:m_left] = beta*ybuf[0:m_left] + (abuf[0:m_left, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm0 );           // ybuf[0:m_left] = zmm0
}

/**
 * The bli_dgemv_n_zen_int_32x3n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 32x3, i.e., 32 rows and 3 columns,
 * and traversing in the n dimension keeping m(32) fixed.
 */
void bli_dgemv_n_zen_int_32x3n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i;

    /**
     * m32_iter: Number of iterations while handling 32 rows at once.
     * m16_iter: Number of iterations while handling 16 rows at once. Can only
     *           be either 0 or 1.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows.
     */
    dim_t m32_iter = m / 32;
    dim_t m_left   = m % 32;
    dim_t m16_iter = m_left / 16;
          m_left   = m_left % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // If m16_iter is non-zero, calculate the result for the 16x2 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m16_iter )
    {
        bli_dgemv_n_zen_int_16x3n_avx512
        (
          conja,
          conjx,
          16,   // Passing m as 16 since we only want to handle 16 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 16*m16_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*m16_iter*incy;
    }

    // If m8_iter is non-zero, calculate the result for the 8x4 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m8_iter )
    {
        bli_dgemv_n_zen_int_8x3n_avx512
        (
          conja,
          conjx,
          8,    // Passing m as 8 since we only want to handle 8 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 8*m8_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*m8_iter*incy;
    }

    // If m_left is non-zero, calculate the result for the remaining rows.
    if ( m_left )
    {
        bli_dgemv_n_zen_int_m_leftx3n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += m_left*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += m_left*incy;
    }

    // intermediate registers
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y
    __m512d zmm4, zmm5, zmm6, zmm7;

    // x
    __m512d zmm26, zmm27, zmm28;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;
    __m512d zmm18, zmm19, zmm20, zmm21;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Loop over the m dimension, i.e., rows of A matrix and the elements
    // in y vector.
    for ( i = 0; i < m32_iter; ++i )
    {
        // Initialize the intermediate registers to zero for every m iteration.
        zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
        zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]
        zmm2 = _mm512_setzero_pd();     // ybuf[16:23]
        zmm3 = _mm512_setzero_pd();     // ybuf[24:31]

        // Move abuf to the beginning of the next block of rows.
        abuf = a + 32*i*rs_a;
        // Initialize xbuf to the beginning of x vector for every iteration.
        xbuf = x;

        // Scale by alpha and broadcast 3 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
        zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

        // Load 32 rows and 3 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
        zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
        zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
        zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
        zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
        zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
        zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
        zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
        zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

        zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
        zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]
        zmm20 = _mm512_loadu_pd( abuf + 16*rs_a + 2*cs_a );     // zmm20 = abuf[16:23, 2]
        zmm21 = _mm512_loadu_pd( abuf + 24*rs_a + 2*cs_a );     // zmm21 = abuf[24:31, 2]

        zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
        zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])
        zmm2 = _mm512_fmadd_pd( zmm28, zmm20, zmm2 );           // ybuf[16:23] += abuf[16:23, 2] * (alpha*x[2])
        zmm3 = _mm512_fmadd_pd( zmm28, zmm21, zmm3 );           // ybuf[24:31] += abuf[24:31, 2] * (alpha*x[2])

        /**
         * If beta = 0, then y vector should not be read, only set.
         * Else, load y, scale by beta, add the intermediate results and store
         * the final result back to y buffer.
         */
        if ( !bli_deq0( *beta ) )
        {
            // Load 32 elements from y vector.
            zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
            zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]
            zmm6 = _mm512_loadu_pd( ybuf + 16*incy );           // zmm6 = ybuf[16:23]
            zmm7 = _mm512_loadu_pd( ybuf + 24*incy );           // zmm7 = ybuf[24:31]

            // Performing the operation y := beta*y + A * (alpha * x).
            // zmm30 = _mm512_set1_pd( *beta );
            zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
            zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
            zmm2 = _mm512_fmadd_pd( zmm30, zmm6, zmm2 );        // ybuf[16:23] = beta*ybuf[16:23] + (abuf[16:23, 0] * (alpha*x[0]))
            zmm3 = _mm512_fmadd_pd( zmm30, zmm7, zmm3 );        // ybuf[24:31] = beta*ybuf[24:31] + (abuf[24:31, 0] * (alpha*x[0]))
        }

        // Store the final result to y buffer.
        _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
        _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1
        _mm512_storeu_pd( ybuf + 16*incy, zmm2 );               // ybuf[16:23] = zmm2
        _mm512_storeu_pd( ybuf + 24*incy, zmm3 );               // ybuf[24:31] = zmm3

        ybuf += 32*incy;    // Move ybuf to the next block of 32 elements.
    }
}

/**
 * The bli_dgemv_n_zen_int_16x3n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 16x3, i.e., 16 rows and 3 columns,
 * and traversing in the n dimension keeping m(16) fixed.
 */
void bli_dgemv_n_zen_int_16x3n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [16, 32).
    dim_t m_left = m % 16;

    // intermediate registers
    __m512d zmm0, zmm1;

    // y
    __m512d zmm4, zmm5;

    // x
    __m512d zmm26, zmm27, zmm28;

    // A
    __m512d zmm10, zmm11;
    __m512d zmm14, zmm15;
    __m512d zmm18, zmm19;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]
    zmm1 = _mm512_setzero_pd();                             // ybuf[ 8:15]

    // Scale by alpha and broadcast 3 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
    zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

    // Load 16 rows and 3 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
    zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
    zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
    zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
    zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

    zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]
    zmm19 = _mm512_loadu_pd( abuf +  8*rs_a + 2*cs_a );     // zmm19 = abuf[ 8:15, 2]

    zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])
    zmm1 = _mm512_fmadd_pd( zmm28, zmm19, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 2] * (alpha*x[2])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 16 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
        zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
        zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
    _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 16*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*incy;

        dim_t m8_iter = m_left / 8;
              m_left  = m_left % 8;

        if ( m8_iter )
        {
            bli_dgemv_n_zen_int_8x3n_avx512
            (
              conja,
              conjx,
              8,    // Passing m as 8 since we only want to handle 8 rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );

            // Move a to the beginning of the next block of rows.
            a    += 8*m8_iter*rs_a;
            // Move ybuf to the beginning of the next block of elements.
            ybuf += 8*m8_iter*incy;
        }

        // If m_left is non-zero, calculate the result for the remaining rows.
        if ( m_left )
        {
            bli_dgemv_n_zen_int_m_leftx3n_avx512
            (
              conja,
              conjx,
              m_left,   // Passing m as m_left since we only want to handle m_left rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );
        }
    }
}

/**
 * The bli_dgemv_n_zen_int_8x3n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 8x3, i.e., 8 rows and 3 columns,
 * and traversing in the n dimension keeping m(8) fixed.
 */
void bli_dgemv_n_zen_int_8x3n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [8, 16).
    dim_t m_left = m % 8;

    // intermediate registers
    __m512d zmm0;

    // y
    __m512d zmm4;

    // x
    __m512d zmm26, zmm27, zmm28;

    // A
    __m512d zmm10;
    __m512d zmm14;
    __m512d zmm18;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]

    // Scale by alpha and broadcast 3 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]
    zmm28 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx)) );  // zmm28 = alpha*x[2*incx]

    // Load 8 rows and 3 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

    zmm18 = _mm512_loadu_pd( abuf +  0*rs_a + 2*cs_a );     // zmm18 = abuf[  0:7, 2]

    zmm0 = _mm512_fmadd_pd( zmm28, zmm18, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 2] * (alpha*x[2])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 8 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 8*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*incy;

        bli_dgemv_n_zen_int_m_leftx3n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );
    }
}

/**
 * The bli_dgemv_n_zen_int_m_leftx4n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of m_leftx3, i.e., m_left rows and 3 columns,
 * and traversing in the n dimension keeping m(m_left) fixed.
 */
void bli_dgemv_n_zen_int_m_leftx3n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m_left,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t cs_a = cs;

    // intermediate registers
    __m512d zmm0;

    // x
    __m512d zmm20, zmm21, zmm22;

    // A
    __m512d zmm10, zmm11, zmm12;

    // y
    __m512d zmm30;

    // beta
    __m512d zmm8;

    // Generate the mask based on the value of m_left.
    __mmask8 m_mask = (1 << m_left) - 1;

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                                     // ybuf[0:m_left]

    // Scale by alpha and broadcast 3 elements from x vector.
    zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );         // zmm20 = alpha*x[0*incx]
    zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );         // zmm21 = alpha*x[1*incx]
    zmm22 = _mm512_set1_pd( *alpha * (*(xbuf + 2*incx) ) );         // zmm22 = alpha*x[2*incx]

    // Load masked rows and 3 columns from A.
    zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );         // zmm10 = abuf[0:m_left, 0]
    zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );         // zmm11 = abuf[0:m_left, 1]
    zmm12 = _mm512_maskz_loadu_pd( m_mask, abuf + 2*cs_a );         // zmm12 = abuf[0:m_left, 2]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 );     // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 );     // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm22, zmm12, zmm0 );     // ybuf[0:m_left] += abuf[0:m_left, 2] * (alpha*x[2])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Broadcast beta to a register.
        zmm8 = _mm512_set1_pd( *beta );                             // zmm8 = beta, beta, beta, beta, ...

        // Load m_left elements from y vector.
        zmm30 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );     // zmm4 = ybuf[0:m_left]

        // Performing the operation y := beta*y + A * (alpha * x).
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm8, zmm30, zmm0 );  // ybuf[0:m_left] = beta*ybuf[0:m_left] + (abuf[0:m_left, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm0 );           // ybuf[0:m_left] = zmm0
}

/**
 * The bli_dgemv_n_zen_int_32x2n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 32x2, i.e., 32 rows and 2 columns,
 * and traversing in the n dimension keeping m(32) fixed.
 */
void bli_dgemv_n_zen_int_32x2n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i;

    /**
     * m32_iter: Number of iterations while handling 32 rows at once.
     * m16_iter: Number of iterations while handling 16 rows at once. Can only
     *           be either 0 or 1.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows.
     */
    dim_t m32_iter = m / 32;
    dim_t m_left   = m % 32;
    dim_t m16_iter = m_left / 16;
          m_left   = m_left % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // If m16_iter is non-zero, calculate the result for the 16x2 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m16_iter )
    {
        bli_dgemv_n_zen_int_16x2n_avx512
        (
          conja,
          conjx,
          16,   // Passing m as 16 since we only want to handle 16 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 16*m16_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*m16_iter*incy;
    }

    // If m8_iter is non-zero, calculate the result for the 8x4 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m8_iter )
    {
        bli_dgemv_n_zen_int_8x2n_avx512
        (
          conja,
          conjx,
          8,    // Passing m as 8 since we only want to handle 8 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 8*m8_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*m8_iter*incy;
    }

    // If m_left is non-zero, calculate the result for the remaining rows.
    if ( m_left )
    {
        bli_dgemv_n_zen_int_m_leftx2n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += m_left*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += m_left*incy;
    }

    // intermediate registers
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y
    __m512d zmm4, zmm5, zmm6, zmm7;

    // x
    __m512d zmm26, zmm27;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;
    __m512d zmm14, zmm15, zmm16, zmm17;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Loop over the m dimension, i.e., rows of A matrix and the elements
    // in y vector.
    for ( i = 0; i < m32_iter; ++i )
    {
        // Initialize the intermediate registers to zero for every m iteration.
        zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
        zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]
        zmm2 = _mm512_setzero_pd();     // ybuf[16:23]
        zmm3 = _mm512_setzero_pd();     // ybuf[24:31]

        // Move abuf to the beginning of the next block of rows.
        abuf = a + 32*i*rs_a;
        // Initialize xbuf to the beginning of x vector for every iteration.
        xbuf = x;

        // Scale by alpha and broadcast 2 elements from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
        zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

        // Load 32 rows and 2 columns from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
        zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
        zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
        zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
        zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

        zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
        zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]
        zmm16 = _mm512_loadu_pd( abuf + 16*rs_a + 1*cs_a );     // zmm16 = abuf[16:23, 1]
        zmm17 = _mm512_loadu_pd( abuf + 24*rs_a + 1*cs_a );     // zmm17 = abuf[24:31, 1]

        zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
        zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])
        zmm2 = _mm512_fmadd_pd( zmm27, zmm16, zmm2 );           // ybuf[16:23] += abuf[16:23, 1] * (alpha*x[1])
        zmm3 = _mm512_fmadd_pd( zmm27, zmm17, zmm3 );           // ybuf[24:31] += abuf[24:31, 1] * (alpha*x[1])

        /**
         * If beta = 0, then y vector should not be read, only set.
         * Else, load y, scale by beta, add the intermediate results and store
         * the final result back to y buffer.
         */
        if ( !bli_deq0( *beta ) )
        {
            // Load 32 elements from y vector.
            zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
            zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]
            zmm6 = _mm512_loadu_pd( ybuf + 16*incy );           // zmm6 = ybuf[16:23]
            zmm7 = _mm512_loadu_pd( ybuf + 24*incy );           // zmm7 = ybuf[24:31]

            // Performing the operation y := beta*y + A * (alpha * x).
            // zmm30 = _mm512_set1_pd( *beta );
            zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
            zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
            zmm2 = _mm512_fmadd_pd( zmm30, zmm6, zmm2 );        // ybuf[16:23] = beta*ybuf[16:23] + (abuf[16:23, 0] * (alpha*x[0]))
            zmm3 = _mm512_fmadd_pd( zmm30, zmm7, zmm3 );        // ybuf[24:31] = beta*ybuf[24:31] + (abuf[24:31, 0] * (alpha*x[0]))
        }

        // Store the final result to y buffer.
        _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
        _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1
        _mm512_storeu_pd( ybuf + 16*incy, zmm2 );               // ybuf[16:23] = zmm2
        _mm512_storeu_pd( ybuf + 24*incy, zmm3 );               // ybuf[24:31] = zmm3

        ybuf += 32*incy;    // Move ybuf to the next block of 32 elements.
    }
}

/**
 * The bli_dgemv_n_zen_int_16x2n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 16x2, i.e., 16 rows and 2 columns,
 * and traversing in the n dimension keeping m(16) fixed.
 */
void bli_dgemv_n_zen_int_16x2n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [16, 32).
    dim_t m_left = m % 16;

    // intermediate registers
    __m512d zmm0, zmm1;

    // y
    __m512d zmm4, zmm5;

    // x
    __m512d zmm26, zmm27;

    // A
    __m512d zmm10, zmm11;
    __m512d zmm14, zmm15;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]
    zmm1 = _mm512_setzero_pd();                             // ybuf[ 8:15]

    // Scale by alpha and broadcast 2 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

    // Load 16 rows and 2 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
    zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
    zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]
    zmm15 = _mm512_loadu_pd( abuf +  8*rs_a + 1*cs_a );     // zmm15 = abuf[ 8:15, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])
    zmm1 = _mm512_fmadd_pd( zmm27, zmm15, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 1] * (alpha*x[1])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 16 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
        zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
        zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
    _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 16*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*incy;

        dim_t m8_iter = m_left / 8;
              m_left  = m_left % 8;

        if ( m8_iter )
        {
            bli_dgemv_n_zen_int_8x2n_avx512
            (
              conja,
              conjx,
              8,    // Passing m as 8 since we only want to handle 8 rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );

            // Move a to the beginning of the next block of rows.
            a    += 8*m8_iter*rs_a;
            // Move ybuf to the beginning of the next block of elements.
            ybuf += 8*m8_iter*incy;
        }

        // If m_left is non-zero, calculate the result for the remaining rows.
        if ( m_left )
        {
            bli_dgemv_n_zen_int_m_leftx2n_avx512
            (
              conja,
              conjx,
              m_left,   // Passing m as m_left since we only want to handle m_left rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );
        }
    }
}

/**
 * The bli_dgemv_n_zen_int_8x2n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 8x2, i.e., 8 rows and 2 columns,
 * and traversing in the n dimension keeping m(8) fixed.
 */
void bli_dgemv_n_zen_int_8x2n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [8, 16).
    dim_t m_left = m % 8;

    // intermediate registers
    __m512d zmm0;

    // y
    __m512d zmm4;

    // x
    __m512d zmm26, zmm27;

    // A
    __m512d zmm10;
    __m512d zmm14;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]

    // Scale by alpha and broadcast 2 elements from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]
    zmm27 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx)) );  // zmm27 = alpha*x[1*incx]

    // Load 8 rows and 2 columns from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

    zmm14 = _mm512_loadu_pd( abuf +  0*rs_a + 1*cs_a );     // zmm14 = abuf[  0:7, 1]

    zmm0 = _mm512_fmadd_pd( zmm27, zmm14, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 1] * (alpha*x[1])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 8 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 8*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*incy;

        bli_dgemv_n_zen_int_m_leftx2n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );
    }
}

/**
 * The bli_dgemv_n_zen_int_m_leftx2n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of m_leftx2, i.e., m_left rows and 2 columns,
 * and traversing in the n dimension keeping m(m_left) fixed.
 */
void bli_dgemv_n_zen_int_m_leftx2n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m_left,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t cs_a = cs;

    // intermediate registers
    __m512d zmm0;

    // x
    __m512d zmm20, zmm21;

    // A
    __m512d zmm10, zmm11;

    // y
    __m512d zmm30;

    // beta
    __m512d zmm8;

    // Generate the mask based on the value of m_left.
    __mmask8 m_mask = (1 << m_left) - 1;

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                                     // ybuf[0:m_left]

    // Scale by alpha and broadcast 2 elements from x vector.
    zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );         // zmm20 = alpha*x[0*incx]
    zmm21 = _mm512_set1_pd( *alpha * (*(xbuf + 1*incx) ) );         // zmm21 = alpha*x[1*incx]

    // Load masked rows and 2 columns from A.
    zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );         // zmm10 = abuf[0:m_left, 0]
    zmm11 = _mm512_maskz_loadu_pd( m_mask, abuf + 1*cs_a );         // zmm11 = abuf[0:m_left, 1]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 );     // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm21, zmm11, zmm0 );     // ybuf[0:m_left] += abuf[0:m_left, 1] * (alpha*x[1])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Broadcast beta to a register.
        zmm8 = _mm512_set1_pd( *beta );                             // zmm8 = beta, beta, beta, beta, ...

        // Load m_left elements from y vector.
        zmm30 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );     // zmm4 = ybuf[0:m_left]

        // Performing the operation y := beta*y + A * (alpha * x).
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm8, zmm30, zmm0 );  // ybuf[0:m_left] = beta*ybuf[0:m_left] + (abuf[0:m_left, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm0 );           // ybuf[0:m_left] = zmm0
}

/**
 * The bli_dgemv_n_zen_int_32x1n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 32x1, i.e., 32 rows and 1 columns,
 * and traversing in the n dimension keeping m(32) fixed.
 */
void bli_dgemv_n_zen_int_32x1n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    dim_t i;

    /**
     * m32_iter: Number of iterations while handling 32 rows at once.
     * m16_iter: Number of iterations while handling 16 rows at once. Can only
     *           be either 0 or 1.
     * m8_iter : Number of iterations while handling 8 rows at once. Can only be
     *           either 0 or 1.
     * m_left  : Number of leftover rows.
     */
    dim_t m32_iter = m / 32;
    dim_t m_left   = m % 32;
    dim_t m16_iter = m_left / 16;
          m_left   = m_left % 16;
    dim_t m8_iter  = m_left / 8;
          m_left   = m_left % 8;

    // If m16_iter is non-zero, calculate the result for the 16x2 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m16_iter )
    {
        bli_dgemv_n_zen_int_16x1n_avx512
        (
          conja,
          conjx,
          16,   // Passing m as 16 since we only want to handle 16 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 16*m16_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*m16_iter*incy;
    }

    // If m8_iter is non-zero, calculate the result for the 8x4 block and
    // update a and ybuf pointers accordingly, to point to the next block.
    if ( m8_iter )
    {
        bli_dgemv_n_zen_int_8x1n_avx512
        (
          conja,
          conjx,
          8,   // Passing m as 8 since we only want to handle 8 rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += 8*m8_iter*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*m8_iter*incy;
    }

    // If m_left is non-zero, calculate the result for the remaining rows.
    if ( m_left )
    {
        bli_dgemv_n_zen_int_m_leftx1n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );

        // Move a to the beginning of the next block of rows.
        a    += m_left*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += m_left*incy;
    }

    // intermediate registers
    __m512d zmm0, zmm1, zmm2, zmm3;

    // y
    __m512d zmm4, zmm5, zmm6, zmm7;

    // x
    __m512d zmm26;

    // A
    __m512d zmm10, zmm11, zmm12, zmm13;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );    // zmm30 = beta, beta, beta, beta, ...

    // Loop over the m dimension, i.e., rows of A matrix and the elements
    // in y vector.
    for ( i = 0; i < m32_iter; ++i )
    {
        // Initialize the intermediate registers to zero for every m iteration.
        zmm0 = _mm512_setzero_pd();     // ybuf[  0:7]
        zmm1 = _mm512_setzero_pd();     // ybuf[ 8:15]
        zmm2 = _mm512_setzero_pd();     // ybuf[16:23]
        zmm3 = _mm512_setzero_pd();     // ybuf[24:31]

        // Move abuf to the beginning of the next block of rows.
        abuf = a + 32*i*rs_a;
        // Initialize xbuf to the beginning of x vector for every iteration.
        xbuf = x;

        // Scale by alpha and broadcast 1 element from x vector.
        zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

        // Load 32 rows and 1 column from A.
        zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]
        zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm11 = abuf[ 8:15, 0]
        zmm12 = _mm512_loadu_pd( abuf + 16*rs_a + 0*cs_a );     // zmm12 = abuf[16:23, 0]
        zmm13 = _mm512_loadu_pd( abuf + 24*rs_a + 0*cs_a );     // zmm13 = abuf[24:31, 0]

        // Performing the operation intermediate_register += A * (alpha * x)
        // and storing the result back to the intermediate registers.
        zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
        zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])
        zmm2 = _mm512_fmadd_pd( zmm26, zmm12, zmm2 );           // ybuf[16:23] += abuf[16:23, 0] * (alpha*x[0])
        zmm3 = _mm512_fmadd_pd( zmm26, zmm13, zmm3 );           // ybuf[24:31] += abuf[24:31, 0] * (alpha*x[0])

        /**
         * If beta = 0, then y vector should not be read, only set.
         * Else, load y, scale by beta, add the intermediate results and store
         * the final result back to y buffer.
         */
        if ( !bli_deq0( *beta ) )
        {
            // Load 32 elements from y vector.
            zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
            zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]
            zmm6 = _mm512_loadu_pd( ybuf + 16*incy );           // zmm6 = ybuf[16:23]
            zmm7 = _mm512_loadu_pd( ybuf + 24*incy );           // zmm7 = ybuf[24:31]

            // Performing the operation y := beta*y + A * (alpha * x).
            // zmm30 = _mm512_set1_pd( *beta );
            zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
            zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
            zmm2 = _mm512_fmadd_pd( zmm30, zmm6, zmm2 );        // ybuf[16:23] = beta*ybuf[16:23] + (abuf[16:23, 0] * (alpha*x[0]))
            zmm3 = _mm512_fmadd_pd( zmm30, zmm7, zmm3 );        // ybuf[24:31] = beta*ybuf[24:31] + (abuf[24:31, 0] * (alpha*x[0]))
        }

        // Store the final result to y buffer.
        _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
        _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1
        _mm512_storeu_pd( ybuf + 16*incy, zmm2 );               // ybuf[16:23] = zmm2
        _mm512_storeu_pd( ybuf + 24*incy, zmm3 );               // ybuf[24:31] = zmm3

        ybuf += 32*incy;    // Move ybuf to the next block of 32 elements.
    }
}

/**
 * The bli_dgemv_n_zen_int_16x1n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 16x1, i.e., 16 rows and 1 columns,
 * and traversing in the n dimension keeping m(16) fixed.
 */
void bli_dgemv_n_zen_int_16x1n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [16, 32).
    dim_t m_left = m % 16;

    // intermediate registers
    __m512d zmm0, zmm1;

    // y
    __m512d zmm4, zmm5;

    // x
    __m512d zmm26;

    // A
    __m512d zmm10, zmm11;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate registers to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]
    zmm1 = _mm512_setzero_pd();                             // ybuf[ 8:15]

    // Scale by alpha and broadcast 1 element from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

    // Load 16 rows and 1 column from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm14 = abuf[  0:7, 1]
    zmm11 = _mm512_loadu_pd( abuf +  8*rs_a + 0*cs_a );     // zmm15 = abuf[ 8:15, 1]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])
    zmm1 = _mm512_fmadd_pd( zmm26, zmm11, zmm1 );           // ybuf[ 8:15] += abuf[ 8:15, 0] * (alpha*x[0])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 16 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]
        zmm5 = _mm512_loadu_pd( ybuf +  8*incy );           // zmm5 = ybuf[ 8:15]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
        zmm1 = _mm512_fmadd_pd( zmm30, zmm5, zmm1 );        // ybuf[ 8:15] = beta*ybuf[ 8:15] + (abuf[ 8:15, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0
    _mm512_storeu_pd( ybuf +  8*incy, zmm1 );               // ybuf[ 8:15] = zmm1

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 16*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 16*incy;

        dim_t m8_iter = m_left / 8;
              m_left  = m_left % 8;

        if ( m8_iter )
        {
            bli_dgemv_n_zen_int_8x1n_avx512
            (
              conja,
              conjx,
              8,    // Passing m as 8 since we only want to handle 8 rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );

            // Move a to the beginning of the next block of rows.
            a    += 8*rs_a;
            // Move ybuf to the beginning of the next block of elements.
            ybuf += 8*incy;
        }

        // If m_left is non-zero, calculate the result for the remaining rows.
        if ( m_left )
        {
            bli_dgemv_n_zen_int_m_leftx1n_avx512
            (
              conja,
              conjx,
              m_left,   // Passing m as m_left since we only want to handle m_left rows.
              n,
              alpha,
              a, rs_a, cs_a,
              x, incx,
              beta,
              ybuf, incy,
              cntx
            );
        }
    }
}

/**
 * The bli_dgemv_n_zen_int_8x1n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of 8x1, i.e., 8 rows and 1 columns,
 * and traversing in the n dimension keeping m(8) fixed.
 */
void bli_dgemv_n_zen_int_8x1n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t rs_a = rs;
    const dim_t cs_a = cs;

    // This kernel will handle sizes where m = [8, 16).
    dim_t m_left = m % 8;

    // intermediate registers
    __m512d zmm0;

    // y
    __m512d zmm4;

    // x
    __m512d zmm26;

    // A
    __m512d zmm10;

    // beta
    // Broadcast beta to a register.
    __m512d zmm30 = _mm512_set1_pd( *beta );                // zmm30 = beta, beta, beta, beta, ...

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                             // ybuf[  0:7]

    // Scale by alpha and broadcast 1 element from x vector.
    zmm26 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx)) );  // zmm26 = alpha*x[0*incx]

    // Load 8 rows and 1 column from A.
    zmm10 = _mm512_loadu_pd( abuf +  0*rs_a + 0*cs_a );     // zmm10 = abuf[  0:7, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_fmadd_pd( zmm26, zmm10, zmm0 );           // ybuf[  0:7] += abuf[  0:7, 0] * (alpha*x[0])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Load 8 elements from y vector.
        zmm4 = _mm512_loadu_pd( ybuf +  0*incy );           // zmm4 = ybuf[  0:7]

        // Performing the operation y := beta*y + A * (alpha * x).
        // zmm30 = _mm512_set1_pd( *beta );
        zmm0 = _mm512_fmadd_pd( zmm30, zmm4, zmm0 );        // ybuf[  0:7] = beta*ybuf[  0:7] + (abuf[  0:7, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_storeu_pd( ybuf +  0*incy, zmm0 );               // ybuf[  0:7] = zmm0

    if ( m_left )
    {
        // Move a to the beginning of the next block of rows.
        a    += 8*rs_a;
        // Move ybuf to the beginning of the next block of elements.
        ybuf += 8*incy;

        bli_dgemv_n_zen_int_m_leftx1n_avx512
        (
          conja,
          conjx,
          m_left,   // Passing m as m_left since we only want to handle m_left rows.
          n,
          alpha,
          a, rs_a, cs_a,
          x, incx,
          beta,
          ybuf, incy,
          cntx
        );
    }
}

/**
 * The bli_dgemv_n_zen_int_m_leftx1n_avx512(...) kernel handles the DGEMV operation
 * by breaking the A matrix into blocks of m_leftx1, i.e., m_left rows and 1 columns,
 * and traversing in the n dimension keeping m(m_left) fixed.
 */
void bli_dgemv_n_zen_int_m_leftx1n_avx512
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m_left,
       dim_t            n,
       double* restrict alpha,
       double* restrict a, inc_t rs, inc_t cs,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    double* restrict abuf = a;
    double* restrict xbuf = x;
    double* restrict ybuf = y;

    const dim_t cs_a = cs;

    // intermediate registers
    __m512d zmm0;

    // x
    __m512d zmm20;

    // A
    __m512d zmm10;

    // y
    __m512d zmm30;

    // beta
    __m512d zmm8;

    // Generate the mask based on the value of m_left.
    __mmask8 m_mask = (1 << m_left) - 1;

    // Initialize the intermediate register to zero.
    zmm0 = _mm512_setzero_pd();                                 // ybuf[0:m_left]

    // Scale by alpha and broadcast 1 element from x vector.
    zmm20 = _mm512_set1_pd( *alpha * (*(xbuf + 0*incx) ) );     // zmm20 = alpha*x[0*incx]

    // Load masked rows and 1 column from A.
    zmm10 = _mm512_maskz_loadu_pd( m_mask, abuf + 0*cs_a );     // zmm10 = abuf[0:m_left, 0]

    // Performing the operation intermediate_register += A * (alpha * x)
    // and storing the result back to the intermediate registers.
    zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm20, zmm10, zmm0 ); // ybuf[0:m_left] += abuf[0:m_left, 0] * (alpha*x[0])

    /**
     * If beta = 0, then y vector should not be read, only set.
     * Else, load y, scale by beta, add the intermediate results and store
     * the final result back to y buffer.
     */
    if ( !bli_deq0( *beta ) )
    {
        // Broadcast beta to a register.
        zmm8 = _mm512_set1_pd( *beta );                             // zmm8 = beta, beta, beta, beta, ...

        // Load m_left elements from y vector.
        zmm30 = _mm512_maskz_loadu_pd( m_mask, ybuf + 0*incy );     // zmm4 = ybuf[0:m_left]

        // Performing the operation y := beta*y + A * (alpha * x).
        zmm0 = _mm512_maskz_fmadd_pd( m_mask, zmm8, zmm30, zmm0 );  // ybuf[0:m_left] = beta*ybuf[0:m_left] + (abuf[0:m_left, 0] * (alpha*x[0]))
    }

    // Store the final result to y buffer.
    _mm512_mask_storeu_pd( ybuf + 0*incy, m_mask, zmm0 );           // ybuf[0:m_left] = zmm0
}
