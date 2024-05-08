/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include "blis.h"
#include "immintrin.h"

#define Z_MR  16
#define Z_NR  4

/* 
    The following API implements the ZGEMM operation specifically for
    inputs A and B with k == 1. It expects the inputs and output to
    support the column-major storage scheme, without any requirement
    to conjugate/transpose any of the operands.

    Design details :
    Kernel dimensions       - 16 x 4
    Loop ordering           - N-loop, followed by M-loop

    The N-Loop will scale B by alpha and presave them on registers
    for its reuse in M-Loop. Thus is blocks 2 * 4(broadcast) registers,
    due to separate real and imaginary components

    Thus the register blocking for the hotspot code-section is as follows :
    Loading A          - 4
    Permuting A        - 4
    alpha * B presave  - 8
    Accumulating C     - 16

    Total              - 32

    Any other register used for miscellaneous computation will not induce
    register dependency explicitly.
*/

err_t bli_zgemm_16x4_avx512_k1_nn
    (
      dim_t  m,
      dim_t  n,
      dim_t  k,
      dcomplex*    alpha,
      dcomplex*    a, const inc_t lda,
      dcomplex*    b, const inc_t ldb,
      dcomplex*    beta,
      dcomplex*    c, const inc_t ldc
    )
{
    // Setting the required variables to choose the right
    // path for computation.
    dim_t m_iter = ( m / Z_MR );
    dim_t n_iter = ( n / Z_NR );

    dim_t m_remainder = ( m % Z_MR );
    dim_t n_remainder = ( n % Z_NR );

    // Setting the alpha and beta scaling components(real and imaginary).
    double alpha_real = alpha->real;
    double alpha_imag = alpha->imag;

    double beta_real = beta->real;
    double beta_imag = beta->imag;

    // Using the predefined enumerated constants to classify beta scaling
    // into one of the below categories.
    dim_t beta_mul_type  = BLIS_MUL_DEFAULT;

    // Setting the appropriate type for beta scaling
    // based on any of the special cases.
    if( beta_imag == 0.0 )
    {
        if( beta_real == 0.0 ) beta_mul_type = BLIS_MUL_ZERO;
        else if( beta_real == 1.0 ) beta_mul_type = BLIS_MUL_ONE;
    }

    // Implementing the GEMM operation, which is as follows :
    // C := beta*C + alpha*A*B.

    // Local pointers for B and C, to be used along the n-loop
    dcomplex* temp_b = b;
    dcomplex* temp_c = c;

    // Main loop along N dimension
    for( dim_t j = 0; j < n_iter; j++ )
    {
        dcomplex* temp_ai = a;
        dcomplex* temp_bj = temp_b;
        dcomplex* temp_cij = temp_c;

        /*
            Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x Z_NR block of B in order to compute the associated
            Z_MR x Z_NR and/or m_remainder x Z_NR block(s) of C. Due to this, the
            associated 1 x Z_NR block of B is scaled with alpha, and stored in registers
            beforehand, to be reused in the main loop or fringe case of m.
        */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m512d a_vec[4], bdcst_real[4], bdcst_imag[4], b_vec[4], temp[4];

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec[0] = _mm512_set1_pd(alpha_real);
        a_vec[1] = _mm512_set1_pd(alpha_imag);

        // Broadcasting real and imag components from B onto separate registers.
        // They are then unpacked to get the interleaved storage format on registers.
        // bdcst_real[0] = R0 R0 R0 R0 ...
        bdcst_real[0] = _mm512_set1_pd(*((double *)(temp_bj)));
        // bdcst_imag[0] = I0 I0 I0 I0 ...
        bdcst_imag[0] = _mm512_set1_pd(*((double *)(temp_bj) + 1));
        // b_vec[0] = R0 I0 R0 I0 ...
        b_vec[0] = _mm512_unpacklo_pd(bdcst_real[0], bdcst_imag[0]);
        // temp[0] = I0 R0 I0 R0 ...
        temp[0] = _mm512_unpacklo_pd(bdcst_imag[0], bdcst_real[0]);

        // bdcst_real[1] = R1 R1 R1 R1 ...
        bdcst_real[1] = _mm512_set1_pd(*((double *)(temp_bj + ldb)));
        // bdcst_imag[1] = I1 I1 I1 I1 ...
        bdcst_imag[1] = _mm512_set1_pd(*((double *)(temp_bj + ldb) + 1));
        // b_vec[1] = R1 I1 R1 I1 ...
        b_vec[1] = _mm512_unpacklo_pd(bdcst_real[1], bdcst_imag[1]);
        // temp[1] = I1 R1 I1 R1 ...
        temp[1] = _mm512_unpacklo_pd(bdcst_imag[1], bdcst_real[1]);

        // Scaling with imag component of alpha
        temp[0] = _mm512_mul_pd(a_vec[1], temp[0]);
        temp[1] = _mm512_mul_pd(a_vec[1], temp[1]);
        // Scaling with real component of alpha and accumulating
        b_vec[0] = _mm512_fmaddsub_pd(a_vec[0], b_vec[0], temp[0]);
        b_vec[1] = _mm512_fmaddsub_pd(a_vec[0], b_vec[1], temp[1]);

        // Continuing the same set of instructions, to load B, unpack
        // them, scale with alpha and store on registers
        // bdcst_real[2] = R2 R2 R2 R2 ...
        bdcst_real[2] = _mm512_set1_pd(*((double *)(temp_bj + 2 * ldb)));
        // bdcst_imag[2] = I2 I2 I2 I2 ...
        bdcst_imag[2] = _mm512_set1_pd(*((double *)(temp_bj + 2 * ldb) + 1));
        // b_vec[2] = R2 I2 R2 I2 ...
        b_vec[2] = _mm512_unpacklo_pd(bdcst_real[2], bdcst_imag[2]);
        // temp[2] = I2 R2 I2 R2 ...
        temp[2] = _mm512_unpacklo_pd(bdcst_imag[2], bdcst_real[2]);

        // bdcst_real[3] = R3 R3 R3 R3 ...
        bdcst_real[3] = _mm512_set1_pd(*((double *)(temp_bj + 3 * ldb)));
        // bdcst_imag[3] = I3 I3 I3 I3 ...
        bdcst_imag[3] = _mm512_set1_pd(*((double *)(temp_bj + 3 * ldb) + 1));
        // b_vec[3] = R3 I3 R3 I3 ...
        b_vec[3] = _mm512_unpacklo_pd(bdcst_real[3], bdcst_imag[3]);
        // temp[3] = I3 R3 I3 R3 ...
        temp[3] = _mm512_unpacklo_pd(bdcst_imag[3], bdcst_real[3]);

        // Scaling with imag component of alpha
        temp[2] = _mm512_mul_pd(a_vec[1], temp[2]);
        temp[3] = _mm512_mul_pd(a_vec[1], temp[3]);
        // Scaling with real component of alpha and accumulating
        b_vec[2] = _mm512_fmaddsub_pd(a_vec[0], b_vec[2], temp[2]);
        b_vec[3] = _mm512_fmaddsub_pd(a_vec[0], b_vec[3], temp[3]);

        // Registers b_vec[0 ... 3] contain alpha scaled B. These
        // are unpacked in order to contain the real and imaginary
        // components of each element in separate registers.
        bdcst_real[0] = _mm512_unpacklo_pd(b_vec[0], b_vec[0]);
        bdcst_real[1] = _mm512_unpacklo_pd(b_vec[1], b_vec[1]);
        bdcst_real[2] = _mm512_unpacklo_pd(b_vec[2], b_vec[2]);
        bdcst_real[3] = _mm512_unpacklo_pd(b_vec[3], b_vec[3]);

        bdcst_imag[0] = _mm512_unpackhi_pd(b_vec[0], b_vec[0]);
        bdcst_imag[1] = _mm512_unpackhi_pd(b_vec[1], b_vec[1]);
        bdcst_imag[2] = _mm512_unpackhi_pd(b_vec[2], b_vec[2]);
        bdcst_imag[3] = _mm512_unpackhi_pd(b_vec[3], b_vec[3]);

        dim_t i = 0;
        dim_t m_rem = m_remainder;
        // Main loop along M dimension.
        for( ; i < m_iter; i++ )
        {
            __m512d a_perm[4], c_vec[16];
            __m512d betaRv, betaIv;

            // Clearing the scratch registers
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();
            c_vec[4] = _mm512_setzero_pd();
            c_vec[5] = _mm512_setzero_pd();
            c_vec[6] = _mm512_setzero_pd();
            c_vec[7] = _mm512_setzero_pd();
            c_vec[8] = _mm512_setzero_pd();
            c_vec[9] = _mm512_setzero_pd();
            c_vec[10] = _mm512_setzero_pd();
            c_vec[11] = _mm512_setzero_pd();
            c_vec[12] = _mm512_setzero_pd();
            c_vec[13] = _mm512_setzero_pd();
            c_vec[14] = _mm512_setzero_pd();
            c_vec[15] = _mm512_setzero_pd();

            // Loading 16 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));
            a_vec[2] = _mm512_loadu_pd((double const*)(temp_ai + 8));
            a_vec[3] = _mm512_loadu_pd((double const*)(temp_ai + 12));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
            a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
            a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);
            c_vec[2] = _mm512_mul_pd(bdcst_imag[0], a_perm[2]);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[0], a_perm[3]);
            c_vec[4] = _mm512_mul_pd(bdcst_imag[1], a_perm[0]);
            c_vec[5] = _mm512_mul_pd(bdcst_imag[1], a_perm[1]);
            c_vec[6] = _mm512_mul_pd(bdcst_imag[1], a_perm[2]);
            c_vec[7] = _mm512_mul_pd(bdcst_imag[1], a_perm[3]);

            c_vec[8] = _mm512_mul_pd(bdcst_imag[2], a_perm[0]);
            c_vec[9] = _mm512_mul_pd(bdcst_imag[2], a_perm[1]);
            c_vec[10] = _mm512_mul_pd(bdcst_imag[2], a_perm[2]);
            c_vec[11] = _mm512_mul_pd(bdcst_imag[2], a_perm[3]);
            c_vec[12] = _mm512_mul_pd(bdcst_imag[3], a_perm[0]);
            c_vec[13] = _mm512_mul_pd(bdcst_imag[3], a_perm[1]);
            c_vec[14] = _mm512_mul_pd(bdcst_imag[3], a_perm[2]);
            c_vec[15] = _mm512_mul_pd(bdcst_imag[3], a_perm[3]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);
            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[2], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[3], c_vec[3]);
            c_vec[4] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[4]);
            c_vec[5] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[1], c_vec[5]);
            c_vec[6] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[2], c_vec[6]);
            c_vec[7] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[3], c_vec[7]);

            c_vec[8] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[0], c_vec[8]);
            c_vec[9] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[1], c_vec[9]);
            c_vec[10] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[2], c_vec[10]);
            c_vec[11] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[3], c_vec[11]);
            c_vec[12] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[0], c_vec[12]);
            c_vec[13] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[1], c_vec[13]);
            c_vec[14] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[2], c_vec[14]);
            c_vec[15] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[3], c_vec[15]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);

                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[8]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[9]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 8), c_vec[10]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 12), c_vec[11]);

                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[12]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[13]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 8), c_vec[14]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 12), c_vec[15]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Adding to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[2]);
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 12));

                    // Adding to alpha*A*B
                    c_vec[4] = _mm512_add_pd(c_vec[4], a_vec[0]);
                    c_vec[5] = _mm512_add_pd(c_vec[5], a_vec[1]);
                    c_vec[6] = _mm512_add_pd(c_vec[6], a_vec[2]);
                    c_vec[7] = _mm512_add_pd(c_vec[7], a_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 12));

                    // Adding to alpha*A*B
                    c_vec[8] = _mm512_add_pd(c_vec[8], a_vec[0]);
                    c_vec[9] = _mm512_add_pd(c_vec[9], a_vec[1]);
                    c_vec[10] = _mm512_add_pd(c_vec[10], a_vec[2]);
                    c_vec[11] = _mm512_add_pd(c_vec[11], a_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[8]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[9]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 8), c_vec[10]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 12), c_vec[11]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 12));

                    // Adding to alpha*A*B
                    c_vec[12] = _mm512_add_pd(c_vec[12], a_vec[0]);
                    c_vec[13] = _mm512_add_pd(c_vec[13], a_vec[1]);
                    c_vec[14] = _mm512_add_pd(c_vec[14], a_vec[2]);
                    c_vec[15] = _mm512_add_pd(c_vec[15], a_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[12]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[13]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 8), c_vec[14]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 12), c_vec[15]);
                    break;

                default :
                    // Loading the real and imag parts of beta
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);
                    c_vec[2] = _mm512_add_pd(a_vec[2], c_vec[2]);
                    c_vec[3] = _mm512_add_pd(a_vec[3], c_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[4] = _mm512_add_pd(a_vec[0], c_vec[4]);
                    c_vec[5] = _mm512_add_pd(a_vec[1], c_vec[5]);
                    c_vec[6] = _mm512_add_pd(a_vec[2], c_vec[6]);
                    c_vec[7] = _mm512_add_pd(a_vec[3], c_vec[7]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[8] = _mm512_add_pd(a_vec[0], c_vec[8]);
                    c_vec[9] = _mm512_add_pd(a_vec[1], c_vec[9]);
                    c_vec[10] = _mm512_add_pd(a_vec[2], c_vec[10]);
                    c_vec[11] = _mm512_add_pd(a_vec[3], c_vec[11]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[8]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[9]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 8), c_vec[10]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 12), c_vec[11]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[12] = _mm512_add_pd(a_vec[0], c_vec[12]);
                    c_vec[13] = _mm512_add_pd(a_vec[1], c_vec[13]);
                    c_vec[14] = _mm512_add_pd(a_vec[2], c_vec[14]);
                    c_vec[15] = _mm512_add_pd(a_vec[3], c_vec[15]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[12]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[13]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 8), c_vec[14]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 12), c_vec[15]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 16;
            temp_ai += 16;
        }

        if( m_rem >= 8 )
        {
            __m512d a_perm[2], c_vec[8];
            __m512d betaRv, betaIv;

            // Clearing the scratch registers
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();
            c_vec[4] = _mm512_setzero_pd();
            c_vec[5] = _mm512_setzero_pd();
            c_vec[6] = _mm512_setzero_pd();
            c_vec[7] = _mm512_setzero_pd();

            // Loading 8 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);
            c_vec[2] = _mm512_mul_pd(bdcst_imag[1], a_perm[0]);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[1], a_perm[1]);

            c_vec[4] = _mm512_mul_pd(bdcst_imag[2], a_perm[0]);
            c_vec[5] = _mm512_mul_pd(bdcst_imag[2], a_perm[1]);
            c_vec[6] = _mm512_mul_pd(bdcst_imag[3], a_perm[0]);
            c_vec[7] = _mm512_mul_pd(bdcst_imag[3], a_perm[1]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);
            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[1], c_vec[3]);

            c_vec[4] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[0], c_vec[4]);
            c_vec[5] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[1], c_vec[5]);
            c_vec[6] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[0], c_vec[6]);
            c_vec[7] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[1], c_vec[7]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);

                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[5]);

                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[7]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Adding it to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));

                    // Adding it to alpha*A*B
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[0]);
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 4));

                    // Adding it to alpha*A*B
                    c_vec[4] = _mm512_add_pd(c_vec[4], a_vec[0]);
                    c_vec[5] = _mm512_add_pd(c_vec[5], a_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[5]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 4));

                    // Adding it to alpha*A*B
                    c_vec[6] = _mm512_add_pd(c_vec[6], a_vec[0]);
                    c_vec[7] = _mm512_add_pd(c_vec[7], a_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[7]);
                    break;

                default :
                    // Loading real and imag components of beta
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[2] = _mm512_add_pd(a_vec[0], c_vec[2]);
                    c_vec[3] = _mm512_add_pd(a_vec[1], c_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[4] = _mm512_add_pd(a_vec[0], c_vec[4]);
                    c_vec[5] = _mm512_add_pd(a_vec[1], c_vec[5]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc + 4), c_vec[5]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[6] = _mm512_add_pd(a_vec[0], c_vec[6]);
                    c_vec[7] = _mm512_add_pd(a_vec[1], c_vec[7]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc + 4), c_vec[7]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 8;
            temp_ai += 8;

            m_rem -= 8;
        }

        if( m_rem >= 4 )
        {
            __m512d a_perm, c_vec[4];
            __m512d betaRv, betaIv;

            // Clearing scratch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();

            // Loading 4 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[1], a_perm);

            c_vec[2] = _mm512_mul_pd(bdcst_imag[2], a_perm);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[3], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[1]);

            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[0], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[0], c_vec[3]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[3]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Adding to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);

                    // Storing the result onto memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[2]);

                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[3]);
                    break;

                default :

                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[1] = _mm512_add_pd(a_vec[0], c_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 2 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[2] = _mm512_add_pd(a_vec[0], c_vec[2]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 2 * ldc), c_vec[2]);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 3 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[3] = _mm512_add_pd(a_vec[0], c_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 3 * ldc), c_vec[3]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 4;
            temp_ai += 4;

            m_rem -= 4;
        }

        if( m_rem > 0 )
        {
            // Setting the mask to load/store the reamining elements
            // Ex : m_rem = 2 => m_mask = ( 1 << 2 * 2 ) - 1
            //                          = 0b0010000 - 1
            //                          = 0b00001111
            // m_rem is multiplied by 2 since it accounts for 2 doubles
            __mmask8 m_mask = m_mask = (1 << 2 * m_rem) - 1;
            __m512d a_perm, c_vec[4];
            __m512d betaRv, betaIv;

            // Clearing the scratch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();

            // Loading the remaining elements from A
            a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[1], a_perm);

            c_vec[2] = _mm512_mul_pd(bdcst_imag[2], a_perm);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[3], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[1]);

            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[2], a_vec[0], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[3], a_vec[0], c_vec[3]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);
                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);
                    _mm512_mask_storeu_pd((double *)(temp_cij + 2 * ldc), m_mask, c_vec[2]);
                    _mm512_mask_storeu_pd((double *)(temp_cij + 3 * ldc), m_mask, c_vec[3]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Adding it to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);

                    // Loading C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 1 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);

                    // Loading C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 2 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 2 * ldc), m_mask, c_vec[2]);

                    // Loading C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 3 * ldc));

                    // Adding it to alpha*A*B
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 3 * ldc), m_mask, c_vec[3]);
                    break;

                default :

                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 1 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[1] = _mm512_add_pd(a_vec[0], c_vec[1]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 2 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[2] = _mm512_add_pd(a_vec[0], c_vec[2]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 2 * ldc), m_mask, c_vec[2]);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 3 * ldc));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[3] = _mm512_add_pd(a_vec[0], c_vec[3]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij + 3 * ldc), m_mask, c_vec[3]);
            }
        }

        // Adjusting the pointers for the next iteration
        temp_b += ldb * Z_NR;
        temp_c += ldc * Z_NR;
    }

    // Fringe case for N
    if( n_remainder >= 2 )
    {
        dcomplex* temp_ai = a;
        dcomplex* temp_bj = temp_b;
        dcomplex* temp_cij = temp_c;

        /*  Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x 2 block of B in order to compute the associated
            Z_MR x 2 and/or m_remainder x 2 block(s) of C. This reusability has been
            exploited, wherein the associated 1 x 2 block of B is scaled with alpha,
            and stored in registers beforehand, to be reused in the main loop or fringe
            case of m.  */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m512d a_vec[4], bdcst_real[2], bdcst_imag[2], b_vec[2], temp[2];

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec[0] = _mm512_set1_pd(alpha_real);
        a_vec[1] = _mm512_set1_pd(alpha_imag);

        // Broadcasting real and imag components from B onto separate registers.
        // They are then unpacked to get the interleaved storage format on registers.
        // bdcst_real[0] = R0 R0 R0 R0 ...
        bdcst_real[0] = _mm512_set1_pd(*((double *)(temp_bj)));
        // bdcst_imag[0] = I0 I0 I0 I0 ...
        bdcst_imag[0] = _mm512_set1_pd(*((double *)(temp_bj) + 1));
        // b_vec[0] = R0 I0 R0 I0 ...
        b_vec[0] = _mm512_unpacklo_pd(bdcst_real[0], bdcst_imag[0]);
        // temp[0] = I0 R0 I0 R0 ...
        temp[0] = _mm512_unpacklo_pd(bdcst_imag[0], bdcst_real[0]);

        // bdcst_real[1] = R1 R1 R1 R1 ...
        bdcst_real[1] = _mm512_set1_pd(*((double *)(temp_bj + ldb)));
        // bdcst_imag[1] = I1 I1 I1 I1 ...
        bdcst_imag[1] = _mm512_set1_pd(*((double *)(temp_bj + ldb) + 1));
        // b_vec[1] = R1 I1 R1 I1 ...
        b_vec[1] = _mm512_unpacklo_pd(bdcst_real[1], bdcst_imag[1]);
        // temp[1] = I1 R1 I1 R1 ...
        temp[1] = _mm512_unpacklo_pd(bdcst_imag[1], bdcst_real[1]);

        // Scaling with imag component of alpha
        temp[0] = _mm512_mul_pd(a_vec[1], temp[0]);
        temp[1] = _mm512_mul_pd(a_vec[1], temp[1]);
        // Scaling with real component of alpha and accumulating
        b_vec[0] = _mm512_fmaddsub_pd(a_vec[0], b_vec[0], temp[0]);
        b_vec[1] = _mm512_fmaddsub_pd(a_vec[0], b_vec[1], temp[1]);

        // Registers b_vec[0 ... 1] contain alpha scaled B. These
        // are unpacked in order to contain the real and imaginary
        // components of each element in separate registers.
        bdcst_real[0] = _mm512_unpacklo_pd(b_vec[0], b_vec[0]);
        bdcst_real[1] = _mm512_unpacklo_pd(b_vec[1], b_vec[1]);

        bdcst_imag[0] = _mm512_unpackhi_pd(b_vec[0], b_vec[0]);
        bdcst_imag[1] = _mm512_unpackhi_pd(b_vec[1], b_vec[1]);

        dim_t i = 0;
        dim_t m_rem = m_remainder;
        // Main loop along M dimension.
        for( ; i < m_iter; i++ )
        {
            __m512d a_perm[4], c_vec[8];
            __m512d betaRv, betaIv;

            // Clearing the scratch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();
            c_vec[4] = _mm512_setzero_pd();
            c_vec[5] = _mm512_setzero_pd();
            c_vec[6] = _mm512_setzero_pd();
            c_vec[7] = _mm512_setzero_pd();

            // Loading 16 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));
            a_vec[2] = _mm512_loadu_pd((double const*)(temp_ai + 8));
            a_vec[3] = _mm512_loadu_pd((double const*)(temp_ai + 12));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
            a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
            a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);
            c_vec[2] = _mm512_mul_pd(bdcst_imag[0], a_perm[2]);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[0], a_perm[3]);
            c_vec[4] = _mm512_mul_pd(bdcst_imag[1], a_perm[0]);
            c_vec[5] = _mm512_mul_pd(bdcst_imag[1], a_perm[1]);
            c_vec[6] = _mm512_mul_pd(bdcst_imag[1], a_perm[2]);
            c_vec[7] = _mm512_mul_pd(bdcst_imag[1], a_perm[3]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);
            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[2], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[3], c_vec[3]);
            c_vec[4] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[4]);
            c_vec[5] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[1], c_vec[5]);
            c_vec[6] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[2], c_vec[6]);
            c_vec[7] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[3], c_vec[7]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Adding C to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[2]);
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 12));

                    c_vec[4] = _mm512_add_pd(c_vec[4], a_vec[0]);
                    c_vec[5] = _mm512_add_pd(c_vec[5], a_vec[1]);
                    c_vec[6] = _mm512_add_pd(c_vec[6], a_vec[2]);
                    c_vec[7] = _mm512_add_pd(c_vec[7], a_vec[3]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);
                    break;

                default :

                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);
                    c_vec[2] = _mm512_add_pd(a_vec[2], c_vec[2]);
                    c_vec[3] = _mm512_add_pd(a_vec[3], c_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);

                    // Registers to load beta(real and imag components)
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 12));

                    // Load C from memory
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with imag component of beta
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    c_vec[4] = _mm512_add_pd(a_vec[0], c_vec[4]);
                    c_vec[5] = _mm512_add_pd(a_vec[1], c_vec[5]);
                    c_vec[6] = _mm512_add_pd(a_vec[2], c_vec[6]);
                    c_vec[7] = _mm512_add_pd(a_vec[3], c_vec[7]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[4]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[5]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 8), c_vec[6]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 12), c_vec[7]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 16;
            temp_ai += 16;
        }

        if( m_rem >= 8 )
        {
            __m512d a_perm[2], c_vec[4];
            __m512d betaRv, betaIv;

            // Clearing out the scratch registers
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();

            // Loading 8 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);
            c_vec[2] = _mm512_mul_pd(bdcst_imag[1], a_perm[0]);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[1], a_perm[1]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);
            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[1], c_vec[3]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);
                    break;

                case BLIS_MUL_ONE :
                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Add C to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);

                    // Store the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));

                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[0]);
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[1]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);
                    break;

                default :

                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);

                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc + 4));

                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[2] = _mm512_add_pd(a_vec[0], c_vec[2]);
                    c_vec[3] = _mm512_add_pd(a_vec[1], c_vec[3]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc + 4), c_vec[3]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 8;
            temp_ai += 8;
            m_rem -= 8;
        }

        if( m_rem >= 4 )
        {
            __m512d a_perm, c_vec[2];
            __m512d betaRv, betaIv;

            // Clearing out sctarch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();

            // Loading 4 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[1], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[1]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Adding it to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);

                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));

                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[0]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);
                    break;

                default :
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);

                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij + 1 * ldc));

                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[1] = _mm512_add_pd(a_vec[0], c_vec[1]);

                    _mm512_storeu_pd((double *)(temp_cij + 1 * ldc), c_vec[1]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 4;
            temp_ai += 4;

            m_rem -= 4;
        }

        if( m_rem > 0 )
        {
            // Setting the mask to load/store remaining elements
            __mmask8 m_mask = m_mask = (1 << 2 * m_rem) - 1;
            __m512d a_perm, c_vec[2];
            __m512d betaRv, betaIv;

            // Clearing out scratch registers
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();

            // Loading remaining elements from A
            a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[1], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[1], a_vec[0], c_vec[1]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);
                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Adding it to alpha*A*B
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);

                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 1 * ldc));

                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[0]);

                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);
                    break;

                default :

                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec[0]);

                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij + 1 * ldc));

                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec[1] = _mm512_add_pd(a_vec[0], c_vec[1]);

                    _mm512_mask_storeu_pd((double *)(temp_cij + 1 * ldc), m_mask, c_vec[1]);
            }
        }

        // Adjusting the pointers accordingly
        temp_b += ldb * 2;
        temp_c += ldc * 2;

        // Updating n_remainder
        n_remainder -= 2;
    }

    if( n_remainder == 1 )
    {
        dcomplex* temp_ai = a;
        dcomplex* temp_bj = temp_b;
        dcomplex* temp_cij = temp_c;

        /*
            Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x 1 block of B in order to compute the associated
            Z_MR x 1 and/or m_remainder x 1 block(s) of C. This reusability has been
            exploited, wherein the associated 1 x 1 block of B is scaled with alpha,
            and stored in registers beforehand, to be reused in the main loop or fringe
            case of m.
        */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m512d a_vec[4], bdcst_real[1], bdcst_imag[1], b_vec[1], temp[1];

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec[0] = _mm512_set1_pd(alpha_real);
        a_vec[1] = _mm512_set1_pd(alpha_imag);

        // Broadcasting real and imag components from B onto separate registers.
        // They are then unpacked to get the interleaved storage format on registers.
        // bdcst_real[0] = R0 R0 R0 R0 ...
        bdcst_real[0] = _mm512_set1_pd(*((double *)(temp_bj)));
        // bdcst_imag[0] = I0 I0 I0 I0 ...
        bdcst_imag[0] = _mm512_set1_pd(*((double *)(temp_bj) + 1));
        // b_vec[0] = R0 I0 R0 I0 ...
        b_vec[0] = _mm512_unpacklo_pd(bdcst_real[0], bdcst_imag[0]);
        // temp[0] = I0 R0 I0 R0 ...
        temp[0] = _mm512_unpacklo_pd(bdcst_imag[0], bdcst_real[0]);

        // Scaling with imag component of alpha
        temp[0] = _mm512_mul_pd(a_vec[1], temp[0]);
        // Scaling with real component of alpha and accumulating
        b_vec[0] = _mm512_fmaddsub_pd(a_vec[0], b_vec[0], temp[0]);

        // Registers b_vec[0] contain alpha scaled B. These
        // are unpacked in order to contain the real and imaginary
        // components of each element in separate registers.
        bdcst_real[0] = _mm512_unpacklo_pd(b_vec[0], b_vec[0]);

        bdcst_imag[0] = _mm512_unpackhi_pd(b_vec[0], b_vec[0]);

        dim_t i = 0;
        dim_t m_rem = m_remainder;
        // Main loop along M dimension.
        for( ; i < m_iter; i++ )
        {
            __m512d a_perm[4], c_vec[4];
            __m512d betaRv, betaIv;

            // Clearing scratch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();
            c_vec[2] = _mm512_setzero_pd();
            c_vec[3] = _mm512_setzero_pd();

            // Loading 16 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));
            a_vec[2] = _mm512_loadu_pd((double const*)(temp_ai + 8));
            a_vec[3] = _mm512_loadu_pd((double const*)(temp_ai + 12));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
            a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
            a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);
            c_vec[2] = _mm512_mul_pd(bdcst_imag[0], a_perm[2]);
            c_vec[3] = _mm512_mul_pd(bdcst_imag[0], a_perm[3]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);
            c_vec[2] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[2], c_vec[2]);
            c_vec[3] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[3], c_vec[3]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading from C
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Adding alpha*A*b to C
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);
                    c_vec[2] = _mm512_add_pd(c_vec[2], a_vec[2]);
                    c_vec[3] = _mm512_add_pd(c_vec[3], a_vec[3]);

                    // Storing to C
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);
                    break;

                default :
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));
                    a_vec[2] = _mm512_loadu_pd((double const*)(temp_cij + 8));
                    a_vec[3] = _mm512_loadu_pd((double const*)(temp_cij + 12));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);
                    a_perm[2] = _mm512_permute_pd(a_vec[2], 0x55);
                    a_perm[3] = _mm512_permute_pd(a_vec[3], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);
                    a_perm[2] = _mm512_mul_pd(betaIv, a_perm[2]);
                    a_perm[3] = _mm512_mul_pd(betaIv, a_perm[3]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);
                    a_vec[2] = _mm512_fmaddsub_pd(betaRv, a_vec[2], a_perm[2]);
                    a_vec[3] = _mm512_fmaddsub_pd(betaRv, a_vec[3], a_perm[3]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);
                    c_vec[2] = _mm512_add_pd(a_vec[2], c_vec[2]);
                    c_vec[3] = _mm512_add_pd(a_vec[3], c_vec[3]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    _mm512_storeu_pd((double *)(temp_cij + 8), c_vec[2]);
                    _mm512_storeu_pd((double *)(temp_cij + 12), c_vec[3]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 16;
            temp_ai += 16;
        }

        if( m_rem >= 8 )
        {
            __m512d a_perm[2], c_vec[2];
            __m512d betaRv, betaIv;

            // Clearing scratch registers for accumalation
            c_vec[0] = _mm512_setzero_pd();
            c_vec[1] = _mm512_setzero_pd();

            // Loading 8 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);
            a_vec[1] = _mm512_loadu_pd((double const*)(temp_ai + 4));

            // Swapping real and imag components, to be used in computation
            a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
            a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

            // Scaling with imag components of alpha*B
            c_vec[0] = _mm512_mul_pd(bdcst_imag[0], a_perm[0]);
            c_vec[1] = _mm512_mul_pd(bdcst_imag[0], a_perm[1]);

            // Scaling with real comp of alpha*B and accumulating
            c_vec[0] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec[0]);
            c_vec[1] = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[1], c_vec[1]);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    break;

                case BLIS_MUL_ONE :
                    // Loading from C
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Adding alpha*A*b to C
                    c_vec[0] = _mm512_add_pd(c_vec[0], a_vec[0]);
                    c_vec[1] = _mm512_add_pd(c_vec[1], a_vec[1]);

                    // Storing to C
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
                    break;

                default :
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));
                    a_vec[1] = _mm512_loadu_pd((double const*)(temp_cij + 4));

                    // Swapping real and imag parts of C for computation
                    a_perm[0] = _mm512_permute_pd(a_vec[0], 0x55);
                    a_perm[1] = _mm512_permute_pd(a_vec[1], 0x55);

                    // Scaling with imag component of beta
                    a_perm[0] = _mm512_mul_pd(betaIv, a_perm[0]);
                    a_perm[1] = _mm512_mul_pd(betaIv, a_perm[1]);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm[0]);
                    a_vec[1] = _mm512_fmaddsub_pd(betaRv, a_vec[1], a_perm[1]);

                    c_vec[0] = _mm512_add_pd(a_vec[0], c_vec[0]);
                    c_vec[1] = _mm512_add_pd(a_vec[1], c_vec[1]);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec[0]);
                    _mm512_storeu_pd((double *)(temp_cij + 4), c_vec[1]);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 8;
            temp_ai += 8;
            m_rem -= 8;
        }

        if( m_rem >= 4 )
        {
            __m512d a_perm, c_vec;
            __m512d betaRv, betaIv;

            // Clearing the scratch register for accumalation
            c_vec = _mm512_setzero_pd();

            // Loading 4 elements from A
            a_vec[0] = _mm512_loadu_pd((double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec = _mm512_mul_pd(bdcst_imag[0], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_storeu_pd((double *)(temp_cij), c_vec);
                    break;

                case BLIS_MUL_ONE :
                    // Loading from C
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Adding alpha*A*b to C
                    c_vec = _mm512_add_pd(c_vec, a_vec[0]);

                    // Storing to C
                    _mm512_storeu_pd((double *)(temp_cij), c_vec);
                    break;

                default :
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_loadu_pd((double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec = _mm512_add_pd(a_vec[0], c_vec);

                    // Storing the result to memory
                    _mm512_storeu_pd((double *)(temp_cij), c_vec);
            }

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += 4;
            temp_ai += 4;

            m_rem -= 4;
        }

        if( m_rem > 0 )
        {
            __mmask8 m_mask = m_mask = (1 << 2 * m_rem) - 1;
            __m512d a_perm, c_vec;
            __m512d betaRv, betaIv;

            // Clearing the scratch register
            c_vec = _mm512_setzero_pd();

            // Loading the remaining elements from A
            a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)temp_ai);

            // Swapping real and imag components, to be used in computation
            a_perm = _mm512_permute_pd(a_vec[0], 0x55);

            // Scaling with imag components of alpha*B
            c_vec = _mm512_mul_pd(bdcst_imag[0], a_perm);

            // Scaling with real comp of alpha*B and accumulating
            c_vec = _mm512_fmaddsub_pd(bdcst_real[0], a_vec[0], c_vec);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    // Storing the result in C.
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec);
                    break;

                case BLIS_MUL_ONE :
                    // Loading from C
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Adding alpha*A*b to C
                    c_vec = _mm512_add_pd(c_vec, a_vec[0]);

                    // Storing to C
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec);
                    break;

                default :
                    betaRv = _mm512_set1_pd(beta_real);
                    betaIv = _mm512_set1_pd(beta_imag);

                    // Load C from memory
                    a_vec[0] = _mm512_maskz_loadu_pd(m_mask, (double const*)(temp_cij));

                    // Swapping real and imag parts of C for computation
                    a_perm = _mm512_permute_pd(a_vec[0], 0x55);

                    // Scaling with imag component of beta
                    a_perm = _mm512_mul_pd(betaIv, a_perm);

                    // Scaling with real component of beta and accumulating
                    a_vec[0] = _mm512_fmaddsub_pd(betaRv, a_vec[0], a_perm);

                    c_vec = _mm512_add_pd(a_vec[0], c_vec);

                    // Storing the result to memory
                    _mm512_mask_storeu_pd((double *)(temp_cij), m_mask, c_vec);
            }
        }
    }

    return BLIS_SUCCESS;
}
