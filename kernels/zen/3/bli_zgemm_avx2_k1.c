/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#define Z_MR  4
#define Z_NR  4

// Macro to be used for beta scaling with 2 loads from C(main loop of m)
#define BETA_SCALING_C_MAIN(reg_0, reg_1, loc) \
\
    /*  Here, a_vec_0 and a_vec_1 are used to load columns of
        length Z_MR from C, with bdcst_0 and bdcst_1 already
        having the real and imaginary parts of beta broadcasted
        onto them. reg_0 and reg_1 are the intermediate registers
        containing the result of alpha*A*B on them. The beta scaling
        and final accumalation is done on these registers for
        storing the corresponding column of C.  */ \
\
    a_vec_0 = _mm256_loadu_pd((double const*)(loc)); \
    a_vec_1 = _mm256_loadu_pd((double const*)(loc + 2)); \
\
    reg_0 = _mm256_fmadd_pd(a_vec_0, bdcst_0, reg_0); \
    reg_1 = _mm256_fmadd_pd(a_vec_1, bdcst_0, reg_1); \
\
    a_vec_0 =  _mm256_permute_pd(a_vec_0, 0x5); \
    a_vec_1 =  _mm256_permute_pd(a_vec_1, 0x5); \
\
    a_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_1); \
    a_vec_1 = _mm256_mul_pd(a_vec_1, bdcst_1); \
\
    reg_0 = _mm256_addsub_pd(reg_0, a_vec_0); \
    reg_1 = _mm256_addsub_pd(reg_1, a_vec_1);

// Macro to be used for beta scaling with 1 load from C(fringe case with m_rem == 1)
#define BETA_SCALING_C_FRINGE(reg_0, loc) \
\
    /*  Here, a_vec_0 is used to load a column of length 2
        from C, with bdcst_0 and bdcst_1 already having the real
        and imaginary parts of beta broadcasted onto them. reg_0
        is the intermediate register containing the result of
        alpha*A*B on it. The beta scaling and final accumalation
        is done on these registers for storing the corresponding
        column of C.  */ \
\
    a_vec_0 = _mm256_loadu_pd((double const*)(loc)); \
\
    reg_0 = _mm256_fmadd_pd(a_vec_0, bdcst_0, reg_0); \
\
    a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5); \
\
    a_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_1); \
\
    reg_0 = _mm256_addsub_pd(reg_0, a_vec_0);

/*  The following API implements the ZGEMM operation specifically for inputs A and B
    with k == 1. It expects the inputs and output to support the column-major storage
    scheme, without any requirement to conjugate/transpose any of the operands.  */

void bli_zgemm_4x4_avx2_k1_nn
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
    // Setting the required variables for choosing the right path
    // to execute the required computation.
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
    int beta_mul_type  = BLIS_MUL_DEFAULT;

    // Setting the appropriate type for beta scaling
    // based on any of the special cases.
    if( beta_imag == 0.0 )
    {
        if( beta_real == 0.0 ) beta_mul_type = BLIS_MUL_ZERO;
        else if( beta_real == 1.0 ) beta_mul_type = BLIS_MUL_ONE;
    }

    // Implementing the GEMM operation, which is as follows :
    // C := beta*C + alpha*A*B.

    // The code structure deals with fringe cases first, followed by the main loop
    // both in the n and m direction.

    // Local pointers for B and C, to be used along the n-loop
    dcomplex* temp_b = b;
    dcomplex* temp_c = c;

    if( ( n_remainder & 0x1 ) == 1 ) // In case of n_remainder being 1 or 3
    {
        // Setting the panel addresses for A, B and C, to be used along m-loop
        dcomplex *temp_ai = a;
        dcomplex *temp_bj = temp_b;
        dcomplex *temp_cij = temp_c;

        /*  Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x 1 block of B in order to compute the associated Z_MR x 1
            and/or m_remainder x 1 block of C. This reusability has been exploited, wherein
            the associated 1 x 1 block of B is scaled with alpha, and stored in
            registers beforehand, to be reused in the main loop or fringe case of m.  */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m256d a_vec_0, a_vec_1;
        __m256d b_vec_0;
        __m256d b_real_0;
        __m256d b_imag_0;
        __m256d bdcst_0, bdcst_1;

        /*  Broadcasting real and imaginary components of elements from B
            and unpacking them to set them in registers in the form :
            { Real_part, Imag_part, Real_part, Imag_part }.

            A total of Z_NR registers are used to store the alpha-scaled B
            for reuse.  */

        b_real_0 = _mm256_broadcast_sd((double const *)(temp_bj));
        b_imag_0 = _mm256_broadcast_sd((double const *)(temp_bj) + 1);
        b_vec_0 = _mm256_unpacklo_pd(b_real_0, b_imag_0);

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec_0 = _mm256_broadcast_sd((double const *)(&alpha_real));
        a_vec_1 = _mm256_broadcast_sd((double const *)(&alpha_imag));

        bdcst_0 = _mm256_unpacklo_pd(b_imag_0, b_real_0);
        bdcst_0 = _mm256_mul_pd(a_vec_1, bdcst_0);
        b_vec_0 = _mm256_fmaddsub_pd(a_vec_0, b_vec_0, bdcst_0);

        // Fringe cases in the m-direction.
        dim_t m_rem = m_remainder;
        if ( ( m_rem & 0x1 ) == 1 )
        {
            // Scratch registers.
            __m256d b_scaled_0, b_perm_0, a_real, a_imag;

            __m128d b_element_0, c_element_0;
            __m128d beta_real_reg, beta_imag_reg, c_perm_0;

            b_scaled_0 = _mm256_setzero_pd();
            b_perm_0 = _mm256_setzero_pd();

            /* Here, only a single element from A is of concern.
               Also, we already have alpha-scaled B available in
               b_vec_0 and b_vec_1. Thus, we could scale these
               registers with the element from A using AVX2 ISA */

            // Broadcasting real and imaginary components from A.

            a_real = _mm256_broadcast_sd((double const *)(temp_ai));
            a_imag = _mm256_broadcast_sd((double const *)(temp_ai) + 1);

            // Obtaining the alpha-scaled B matrix
            b_scaled_0 = b_vec_0;
            b_perm_0 = _mm256_permute_pd(b_scaled_0, 0x5);

            b_perm_0 = _mm256_mul_pd(b_perm_0, a_imag);
            b_scaled_0 = _mm256_fmaddsub_pd(b_scaled_0, a_real, b_perm_0);

            c_element_0 = _mm256_castpd256_pd128(b_scaled_0);

            // Clearing out the upper lanes of 256 bit registers to avoid
            // the transition penalty
            _mm256_zeroupper();

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));
                    c_element_0 = _mm_add_pd(c_element_0, b_element_0);
                    break;

                default :
                    // Broadcast beta real and imaginary part and scale with C.
                    beta_real_reg = _mm_loaddup_pd((double const*)beta);
                    beta_imag_reg = _mm_loaddup_pd((double const*)beta + 1);

                    // Load C onto registers
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));

                    // Shuffle for the compute with imgarinary part scaling
                    c_perm_0 = _mm_shuffle_pd(b_element_0, b_element_0, 0x01);

                    c_perm_0 = _mm_mul_pd(beta_imag_reg, c_perm_0);

                    b_element_0 = _mm_mul_pd(beta_real_reg, b_element_0);
                    // Compute beta-scaled C
                    b_element_0 = _mm_addsub_pd(b_element_0, c_perm_0);
                    // Add to intermediate reg storing alpha*A*B
                    c_element_0 = _mm_add_pd(b_element_0, c_element_0);
            }

            // Storing the result in C.
            _mm_storeu_pd((double *)(temp_cij), c_element_0);

            // We need to restore the upper lanes of the registers b_vec_0, b_vec_1,
            // b_vec_2 and b_vec_3
            // They need to contain the alpha scaled B, to be reused in the main loop for m
            b_element_0 = _mm256_castpd256_pd128(b_vec_0);
            b_vec_0 = _mm256_insertf128_pd(b_vec_0, b_element_0, 0x01);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 1;
            temp_ai += 1;

            m_rem -= 1;
        }

        if( m_rem == 2 )
        {
            // Scratch registers.
            __m256d c_vec_0;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();

            // Loading a vector from A with 2 elements.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_FRINGE(c_vec_0, temp_cij);
            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 2;
            temp_ai += 2;

            m_rem -= 2;
        }

        // Main loop along M dimension.
        for( dim_t i = 0; i < m_iter; i++ )
        {
            // Scratch registers
            __m256d c_vec_0, c_vec_1;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();
            c_vec_1 = _mm256_setzero_pd();

            // Prefetching the block of C to be used for computation.
            _mm_prefetch((char const*)(temp_cij), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*2), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*3), _MM_HINT_T0);

            // Loading vectors from A with Z_MR elements in total.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));
            a_vec_1 = _mm256_loadu_pd((double const *)(temp_ai + 2));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_1 = _mm256_mul_pd(a_vec_1, bdcst_0);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);
            c_vec_1 = _mm256_fmaddsub_pd(a_vec_1, bdcst_0, c_vec_1);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + 2));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);
                    c_vec_1 = _mm256_add_pd(c_vec_1, a_vec_1);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_MAIN(c_vec_0, c_vec_1, temp_cij);

            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);
            _mm256_storeu_pd((double *)(temp_cij + 2), c_vec_1);

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += Z_MR;
            temp_ai += Z_MR;

        }

        temp_b += ldb;
        temp_c += ldc;

        n_remainder -= 1;
    }

    if( n_remainder == 2 )
    {
        // Setting the panel addresses for A B, and C, to be used along m-loop
        dcomplex *temp_ai = a;
        dcomplex *temp_bj = temp_b;
        dcomplex *temp_cij = temp_c;

        /*  Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x 2 block of B in order to compute the associated Z_MR x 2
            and/or m_remainder x 2 block of C. This reusability has been exploited, wherein
            the associated 1 x 2 block of B is scaled with alpha, and stored in
            registers beforehand, to be reused in the main loop or fringe case of m.  */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m256d a_vec_0, a_vec_1;
        __m256d b_vec_0, b_vec_1;
        __m256d b_real_0, b_real_1;
        __m256d b_imag_0, b_imag_1;
        __m256d bdcst_0, bdcst_1;

        /*  Broadcasting real and imaginary components of elements from B
            and unpacking them to set them in registers in the form :
            { Real_part, Imag_part, Real_part, Imag_part }.

            A total of Z_NR registers are used to store the alpha-scaled B
            for reuse.  */

        b_real_0 = _mm256_broadcast_sd((double const *)(temp_bj));
        b_imag_0 = _mm256_broadcast_sd((double const *)(temp_bj) + 1);
        b_vec_0 = _mm256_unpacklo_pd(b_real_0, b_imag_0);

        b_real_1 = _mm256_broadcast_sd((double const *)(temp_bj + ldb));
        b_imag_1 = _mm256_broadcast_sd((double const *)(temp_bj + ldb) + 1);
        b_vec_1 = _mm256_unpacklo_pd(b_real_1, b_imag_1);

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec_0 = _mm256_broadcast_sd((double const *)(&alpha_real));
        a_vec_1 = _mm256_broadcast_sd((double const *)(&alpha_imag));

        bdcst_0 = _mm256_unpacklo_pd(b_imag_0, b_real_0);
        bdcst_1 = _mm256_unpacklo_pd(b_imag_1, b_real_1);
        bdcst_0 = _mm256_mul_pd(a_vec_1, bdcst_0);
        bdcst_1 = _mm256_mul_pd(a_vec_1, bdcst_1);
        b_vec_0 = _mm256_fmaddsub_pd(a_vec_0, b_vec_0, bdcst_0);
        b_vec_1 = _mm256_fmaddsub_pd(a_vec_0, b_vec_1, bdcst_1);

        // Fringe cases in the m-direction.
        dim_t m_rem = m_remainder;
        if ( ( m_rem & 0x1 ) == 1 )
        {
            // Scratch registers.
            __m256d b_scaled_0, b_perm_0, a_real, a_imag;

            __m128d b_element_0, b_element_1, c_element_0, c_element_1;
            __m128d beta_real_reg, beta_imag_reg, c_perm_0, c_perm_1;

            b_scaled_0 = _mm256_setzero_pd();
            b_perm_0 = _mm256_setzero_pd();

            /* Here, only a single element from A is of concern.
               Also, we already have alpha-scaled B available in
               b_vec_0 and b_vec_1. Thus, we could scale these
               registers with the element from A using AVX2 ISA */

            // Broadcasting real and imaginary components from A.

            a_real = _mm256_broadcast_sd((double const *)(temp_ai));
            a_imag = _mm256_broadcast_sd((double const *)(temp_ai) + 1);

            // Obtaining the alpha-scaled B matrix

            b_scaled_0 = _mm256_permute2f128_pd(b_vec_0, b_vec_1, 0x20);
            b_perm_0 = _mm256_permute_pd(b_scaled_0, 0x5);

            b_perm_0 = _mm256_mul_pd(b_perm_0, a_imag);
            b_scaled_0 = _mm256_fmaddsub_pd(b_scaled_0, a_real, b_perm_0);

            c_element_0 = _mm256_castpd256_pd128(b_scaled_0);
            c_element_1 = _mm256_extractf128_pd(b_scaled_0, 0x01);

            // Clearing out the upper lanes of 256 bit registers to avoid
            // the transition penalty
            _mm256_zeroupper();

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));
                    c_element_0 = _mm_add_pd(c_element_0, b_element_0);

                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc));
                    c_element_1 = _mm_add_pd(c_element_1, b_element_1);
                    break;

                default :
                    // Broadcast beta real and imaginary part and scale with C.
                    beta_real_reg = _mm_loaddup_pd((double const*)beta);
                    beta_imag_reg = _mm_loaddup_pd((double const*)beta + 1);

                    // Load C onto registers
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));
                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc));

                    // Shuffle for the compute with imgarinary part scaling
                    c_perm_0 = _mm_shuffle_pd(b_element_0, b_element_0, 0x01);
                    c_perm_1 = _mm_shuffle_pd(b_element_1, b_element_1, 0x01);

                    c_perm_0 = _mm_mul_pd(beta_imag_reg, c_perm_0);
                    c_perm_1 = _mm_mul_pd(beta_imag_reg, c_perm_1);

                    b_element_0 = _mm_mul_pd(beta_real_reg, b_element_0);
                    b_element_1 = _mm_mul_pd(beta_real_reg, b_element_1);

                    // Compute beta-scaled C
                    b_element_0 = _mm_addsub_pd(b_element_0, c_perm_0);
                    b_element_1 = _mm_addsub_pd(b_element_1, c_perm_1);

                    // Add to intermediate reg storing alpha*A*B
                    c_element_0 = _mm_add_pd(b_element_0, c_element_0);
                    c_element_1 = _mm_add_pd(b_element_1, c_element_1);
            }

            // Storing the result in C.
            _mm_storeu_pd((double *)(temp_cij), c_element_0);
            _mm_storeu_pd((double *)(temp_cij + ldc), c_element_1);

            // We need to restore the upper lanes of the registers b_vec_0, b_vec_1,
            // b_vec_2 and b_vec_3
            // They need to contain the alpha scaled B, to be reused in the main loop for m
            b_element_0 = _mm256_castpd256_pd128(b_vec_0);
            b_element_1 = _mm256_extractf128_pd(b_vec_1, 0x00);
            b_vec_0 = _mm256_insertf128_pd(b_vec_0, b_element_0, 0x01);
            b_vec_1 = _mm256_insertf128_pd(b_vec_1, b_element_1, 0x01);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 1;
            temp_ai += 1;

            m_rem -= 1;
        }

        if( m_rem == 2 )
        {
            // Scratch registers.
            __m256d c_vec_0, c_vec_2;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();
            c_vec_2 = _mm256_setzero_pd();

            // Loading a vector from A with 2 elements.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_2 = _mm256_mul_pd(a_vec_0, bdcst_1);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);
            c_vec_2 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_2);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc));
                    c_vec_2 = _mm256_add_pd(c_vec_2, a_vec_0);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_FRINGE(c_vec_0, temp_cij);
                    BETA_SCALING_C_FRINGE(c_vec_2, temp_cij + ldc);

            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);
            _mm256_storeu_pd((double *)(temp_cij + ldc), c_vec_2);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 2;
            temp_ai += 2;

            m_rem -= 2;
        }

        // Main loop along M dimension.
        for( dim_t i = 0; i < m_iter; i++ )
        {
            // Scratch registers
            __m256d c_vec_0, c_vec_1, c_vec_2, c_vec_3;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();
            c_vec_1 = _mm256_setzero_pd();
            c_vec_2 = _mm256_setzero_pd();
            c_vec_3 = _mm256_setzero_pd();

            // Prefetching the block of C to be used for computation.
            _mm_prefetch((char const*)(temp_cij), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*2), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*3), _MM_HINT_T0);

            // Loading vectors from A with Z_MR elements in total.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));
            a_vec_1 = _mm256_loadu_pd((double const *)(temp_ai + 2));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_1 = _mm256_mul_pd(a_vec_1, bdcst_0);
            c_vec_2 = _mm256_mul_pd(a_vec_0, bdcst_1);
            c_vec_3 = _mm256_mul_pd(a_vec_1, bdcst_1);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);
            c_vec_1 = _mm256_fmaddsub_pd(a_vec_1, bdcst_0, c_vec_1);
            c_vec_2 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_2);
            c_vec_3 = _mm256_fmaddsub_pd(a_vec_1, bdcst_1, c_vec_3);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + 2));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);
                    c_vec_1 = _mm256_add_pd(c_vec_1, a_vec_1);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + ldc + 2));
                    c_vec_2 = _mm256_add_pd(c_vec_2, a_vec_0);
                    c_vec_3 = _mm256_add_pd(c_vec_3, a_vec_1);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_MAIN(c_vec_0, c_vec_1, temp_cij);
                    BETA_SCALING_C_MAIN(c_vec_2, c_vec_3, temp_cij + ldc);

            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);
            _mm256_storeu_pd((double *)(temp_cij + 2), c_vec_1);

            _mm256_storeu_pd((double *)(temp_cij + ldc), c_vec_2);
            _mm256_storeu_pd((double *)(temp_cij + ldc + 2), c_vec_3);

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += Z_MR;
            temp_ai += Z_MR;

        }

        temp_b += ldb*2;
        temp_c += ldc*2;

        n_remainder -= 2;
    }

    // Main loop along N dimension
    for( dim_t j = 0; j < n_iter; j++ )
    {
        dcomplex* temp_bj = temp_b + j * ldb * Z_NR;
        dcomplex* temp_ai = a;
        dcomplex* temp_cij = temp_c + j * ldc * Z_NR;

        /*  Multiple blocks of Z_MR x 1(main loop for m) and/or m_remainder x 1 block(s)
            of A use the same 1 x Z_NR block of B in order to compute the associated
            Z_MR x Z_NR and/or m_remainder x Z_NR block(s) of C. This reusability has been
            exploited, wherein the associated 1 x Z_NR block of B is scaled with alpha,
            and stored in registers beforehand, to be reused in the main loop or fringe
            case of m.  */

        // Intermediate registers used for alpha scaling the block of B and storing.
        __m256d a_vec_0, a_vec_1;
        __m256d b_vec_0, b_vec_1, b_vec_2, b_vec_3;
        __m256d b_real_0, b_real_1, b_real_2, b_real_3;
        __m256d b_imag_0, b_imag_1, b_imag_2, b_imag_3;
        __m256d bdcst_0, bdcst_1;

        /*  Broadcasting real and imaginary components of elements from B
            and unpacking them to set them in registers in the form :
            { Real_part, Imag_part, Real_part, Imag_part }.

            A total of Z_NR registers are used to store the alpha-scaled B
            for reuse.  */

        b_real_0 = _mm256_broadcast_sd((double const *)(temp_bj));
        b_imag_0 = _mm256_broadcast_sd((double const *)(temp_bj) + 1);
        b_vec_0 = _mm256_unpacklo_pd(b_real_0, b_imag_0);

        b_real_1 = _mm256_broadcast_sd((double const *)(temp_bj + ldb));
        b_imag_1 = _mm256_broadcast_sd((double const *)(temp_bj + ldb) + 1);
        b_vec_1 = _mm256_unpacklo_pd(b_real_1, b_imag_1);

        b_real_2 = _mm256_broadcast_sd((double const *)(temp_bj + ldb*2));
        b_imag_2 = _mm256_broadcast_sd((double const *)(temp_bj + ldb*2) + 1);
        b_vec_2 = _mm256_unpacklo_pd(b_real_2, b_imag_2);

        b_real_3 = _mm256_broadcast_sd((double const *)(temp_bj + ldb*3));
        b_imag_3 = _mm256_broadcast_sd((double const *)(temp_bj + ldb*3) + 1);
        b_vec_3 = _mm256_unpacklo_pd(b_real_3, b_imag_3);

        // Broadcast elements from alpha, and exhibit the compute for complex scaling.
        a_vec_0 = _mm256_broadcast_sd((double const *)(&alpha_real));
        a_vec_1 = _mm256_broadcast_sd((double const *)(&alpha_imag));

        bdcst_0 = _mm256_unpacklo_pd(b_imag_0, b_real_0);
        bdcst_1 = _mm256_unpacklo_pd(b_imag_1, b_real_1);
        bdcst_0 = _mm256_mul_pd(a_vec_1, bdcst_0);
        bdcst_1 = _mm256_mul_pd(a_vec_1, bdcst_1);
        b_vec_0 = _mm256_fmaddsub_pd(a_vec_0, b_vec_0, bdcst_0);
        b_vec_1 = _mm256_fmaddsub_pd(a_vec_0, b_vec_1, bdcst_1);

        bdcst_0 = _mm256_unpacklo_pd(b_imag_2, b_real_2);
        bdcst_1 = _mm256_unpacklo_pd(b_imag_3, b_real_3);
        bdcst_0 = _mm256_mul_pd(a_vec_1, bdcst_0);
        bdcst_1 = _mm256_mul_pd(a_vec_1, bdcst_1);
        b_vec_2 = _mm256_fmaddsub_pd(a_vec_0, b_vec_2, bdcst_0);
        b_vec_3 = _mm256_fmaddsub_pd(a_vec_0, b_vec_3, bdcst_1);

        // Fringe cases in the m-direction.
        dim_t m_rem = m_remainder;
        if ( ( m_rem & 0x1 ) == 1 )
        {
            // Scratch registers.
            __m256d b_scaled_0, b_perm_0, a_real, a_imag;

            __m128d b_element_0, b_element_1;
            __m128d c_element_0, c_element_1, c_element_2, c_element_3;
            __m128d beta_real_reg, beta_imag_reg, c_perm_0, c_perm_1;

            b_scaled_0 = _mm256_setzero_pd();
            b_perm_0 = _mm256_setzero_pd();

            /* Here, only a single element from A is of concern.
               Also, we already have alpha-scaled B available in
               b_vec_0 and b_vec_1. Thus, we could scale these
               registers with the element from A using AVX2 ISA */

            // Broadcasting real and imaginary components from A.

            a_real = _mm256_broadcast_sd((double const *)(temp_ai));
            a_imag = _mm256_broadcast_sd((double const *)(temp_ai) + 1);

            // Obtaining the alpha-scaled B matrix

            b_scaled_0 = _mm256_permute2f128_pd(b_vec_0, b_vec_1, 0x20);
            b_perm_0 = _mm256_permute_pd(b_scaled_0, 0x5);

            b_perm_0 = _mm256_mul_pd(b_perm_0, a_imag);
            b_scaled_0 = _mm256_fmaddsub_pd(b_scaled_0, a_real, b_perm_0);

            c_element_0 = _mm256_castpd256_pd128(b_scaled_0);
            c_element_1 = _mm256_extractf128_pd(b_scaled_0, 0x01);

            b_scaled_0 = _mm256_permute2f128_pd(b_vec_2, b_vec_3, 0x20);
            b_perm_0 = _mm256_permute_pd(b_scaled_0, 0x5);

            b_perm_0 = _mm256_mul_pd(b_perm_0, a_imag);
            b_scaled_0 = _mm256_fmaddsub_pd(b_scaled_0, a_real, b_perm_0);

            c_element_2 = _mm256_castpd256_pd128(b_scaled_0);
            c_element_3 = _mm256_extractf128_pd(b_scaled_0, 0x01);

            // Clearing out the upper lanes of 256 bit registers to avoid
            // the transition penalty
            _mm256_zeroupper();

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));
                    c_element_0 = _mm_add_pd(c_element_0, b_element_0);

                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc));
                    c_element_1 = _mm_add_pd(c_element_1, b_element_1);

                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij + ldc*2));
                    c_element_2 = _mm_add_pd(c_element_2, b_element_0);

                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc*3));
                    c_element_3 = _mm_add_pd(c_element_3, b_element_1);
                    break;

                default :
                    // Broadcast beta real and imaginary part and scale with C.
                    beta_real_reg = _mm_loaddup_pd((double const*)beta);
                    beta_imag_reg = _mm_loaddup_pd((double const*)beta + 1);

                    // Load C onto registers
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij));
                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc));

                    // Shuffle for the compute with imgarinary part scaling
                    c_perm_0 = _mm_shuffle_pd(b_element_0, b_element_0, 0x01);
                    c_perm_1 = _mm_shuffle_pd(b_element_1, b_element_1, 0x01);

                    c_perm_0 = _mm_mul_pd(beta_imag_reg, c_perm_0);
                    c_perm_1 = _mm_mul_pd(beta_imag_reg, c_perm_1);

                    b_element_0 = _mm_mul_pd(beta_real_reg, b_element_0);
                    b_element_1 = _mm_mul_pd(beta_real_reg, b_element_1);

                    // Compute beta-scaled C
                    b_element_0 = _mm_addsub_pd(b_element_0, c_perm_0);
                    b_element_1 = _mm_addsub_pd(b_element_1, c_perm_1);

                    // Add to intermediate reg storing alpha*A*B
                    c_element_0 = _mm_add_pd(b_element_0, c_element_0);
                    c_element_1 = _mm_add_pd(b_element_1, c_element_1);

                    // Load C onto registers
                    b_element_0 = _mm_loadu_pd((double const*)(temp_cij + ldc*2));
                    b_element_1 = _mm_loadu_pd((double const*)(temp_cij + ldc*3));

                    // Shuffle for the compute with imgarinary part scaling
                    c_perm_0 = _mm_shuffle_pd(b_element_0, b_element_0, 0x01);
                    c_perm_1 = _mm_shuffle_pd(b_element_1, b_element_1, 0x01);

                    c_perm_0 = _mm_mul_pd(beta_imag_reg, c_perm_0);
                    c_perm_1 = _mm_mul_pd(beta_imag_reg, c_perm_1);

                    b_element_0 = _mm_mul_pd(beta_real_reg, b_element_0);
                    b_element_1 = _mm_mul_pd(beta_real_reg, b_element_1);

                    // Compute beta-scaled C
                    b_element_0 = _mm_addsub_pd(b_element_0, c_perm_0);
                    b_element_1 = _mm_addsub_pd(b_element_1, c_perm_1);

                    // Add to intermediate reg storing alpha*A*B
                    c_element_2 = _mm_add_pd(b_element_0, c_element_2);
                    c_element_3 = _mm_add_pd(b_element_1, c_element_3);
            }

            // Storing the result in C.
            _mm_storeu_pd((double *)(temp_cij), c_element_0);
            _mm_storeu_pd((double *)(temp_cij + ldc), c_element_1);
            _mm_storeu_pd((double *)(temp_cij + ldc*2), c_element_2);
            _mm_storeu_pd((double *)(temp_cij + ldc*3), c_element_3);

            // We need to restore the upper lanes of the registers b_vec_0, b_vec_1,
            // b_vec_2 and b_vec_3
            // They need to contain the alpha scaled B, to be reused in the main loop for m
            b_element_0 = _mm256_castpd256_pd128(b_vec_0);
            b_element_1 = _mm256_castpd256_pd128(b_vec_1);
            b_vec_0 = _mm256_insertf128_pd(b_vec_0, b_element_0, 0x01);
            b_vec_1 = _mm256_insertf128_pd(b_vec_1, b_element_1, 0x01);

            b_element_0 = _mm256_castpd256_pd128(b_vec_2);
            b_element_1 = _mm256_castpd256_pd128(b_vec_3);
            b_vec_2 = _mm256_insertf128_pd(b_vec_2, b_element_0, 0x01);
            b_vec_3 = _mm256_insertf128_pd(b_vec_3, b_element_1, 0x01);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 1;
            temp_ai += 1;

            m_rem -= 1;
        }

        if( m_rem >= 2 )
        {
            // Scratch registers.
            __m256d c_vec_0, c_vec_2, c_vec_4, c_vec_6;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();
            c_vec_2 = _mm256_setzero_pd();
            c_vec_4 = _mm256_setzero_pd();
            c_vec_6 = _mm256_setzero_pd();

            // Loading a vector from A with 2 elements.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_2 = _mm256_mul_pd(a_vec_0, bdcst_1);

            bdcst_0 = _mm256_unpackhi_pd(b_vec_2, b_vec_2);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_3, b_vec_3);
            c_vec_4 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_6 = _mm256_mul_pd(a_vec_0, bdcst_1);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);
            c_vec_2 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_2);

            bdcst_0 = _mm256_unpacklo_pd(b_vec_2, b_vec_2);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_3, b_vec_3);
            c_vec_4 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_4);
            c_vec_6 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_6);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc));
                    c_vec_2 = _mm256_add_pd(c_vec_2, a_vec_0);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc*2));
                    c_vec_4 = _mm256_add_pd(c_vec_4, a_vec_0);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc*3));
                    c_vec_6 = _mm256_add_pd(c_vec_6, a_vec_0);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_FRINGE(c_vec_0, temp_cij);
                    BETA_SCALING_C_FRINGE(c_vec_2, temp_cij + ldc);
                    BETA_SCALING_C_FRINGE(c_vec_4, temp_cij + ldc*2);
                    BETA_SCALING_C_FRINGE(c_vec_6, temp_cij + ldc*3);

            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);
            _mm256_storeu_pd((double *)(temp_cij + ldc), c_vec_2);
            _mm256_storeu_pd((double *)(temp_cij + ldc*2), c_vec_4);
            _mm256_storeu_pd((double *)(temp_cij + ldc*3), c_vec_6);

            // Adjusting the addresses of A and C for the next block.
            temp_cij += 2;
            temp_ai += 2;

            m_rem -= 2;
        }

        // Main loop along M dimension.
        for( dim_t i = 0; i < m_iter; i++ )
        {
            // Scratch registers.
            __m256d c_vec_0, c_vec_1, c_vec_2, c_vec_3;
            __m256d c_vec_4, c_vec_5, c_vec_6, c_vec_7;

            a_vec_0 = _mm256_setzero_pd();
            a_vec_1 = _mm256_setzero_pd();
            bdcst_0 = _mm256_setzero_pd();
            bdcst_1 = _mm256_setzero_pd();
            c_vec_0 = _mm256_setzero_pd();
            c_vec_1 = _mm256_setzero_pd();
            c_vec_2 = _mm256_setzero_pd();
            c_vec_3 = _mm256_setzero_pd();
            c_vec_4 = _mm256_setzero_pd();
            c_vec_5 = _mm256_setzero_pd();
            c_vec_6 = _mm256_setzero_pd();
            c_vec_7 = _mm256_setzero_pd();

            _mm_prefetch((char const*)(temp_cij), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*2), _MM_HINT_T0);
            _mm_prefetch((char const*)(temp_cij + ldc*3), _MM_HINT_T0);

            // Loading vectors from A with Z_MR elements in total.
            a_vec_0 = _mm256_loadu_pd((double const *)(temp_ai));
            a_vec_1 = _mm256_loadu_pd((double const *)(temp_ai + 2));

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with imaginary components of elements from B.
            bdcst_0 = _mm256_unpackhi_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_1 = _mm256_mul_pd(a_vec_1, bdcst_0);
            c_vec_2 = _mm256_mul_pd(a_vec_0, bdcst_1);
            c_vec_3 = _mm256_mul_pd(a_vec_1, bdcst_1);

            bdcst_0 = _mm256_unpackhi_pd(b_vec_2, b_vec_2);
            bdcst_1 = _mm256_unpackhi_pd(b_vec_3, b_vec_3);
            c_vec_4 = _mm256_mul_pd(a_vec_0, bdcst_0);
            c_vec_5 = _mm256_mul_pd(a_vec_1, bdcst_0);
            c_vec_6 = _mm256_mul_pd(a_vec_0, bdcst_1);
            c_vec_7 = _mm256_mul_pd(a_vec_1, bdcst_1);

            a_vec_0 = _mm256_permute_pd(a_vec_0, 0x5);
            a_vec_1 = _mm256_permute_pd(a_vec_1, 0x5);

            // Scaling with real components of elements from B.
            bdcst_0 = _mm256_unpacklo_pd(b_vec_0, b_vec_0);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_1, b_vec_1);
            c_vec_0 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_0);
            c_vec_1 = _mm256_fmaddsub_pd(a_vec_1, bdcst_0, c_vec_1);
            c_vec_2 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_2);
            c_vec_3 = _mm256_fmaddsub_pd(a_vec_1, bdcst_1, c_vec_3);

            bdcst_0 = _mm256_unpacklo_pd(b_vec_2, b_vec_2);
            bdcst_1 = _mm256_unpacklo_pd(b_vec_3, b_vec_3);
            c_vec_4 = _mm256_fmaddsub_pd(a_vec_0, bdcst_0, c_vec_4);
            c_vec_5 = _mm256_fmaddsub_pd(a_vec_1, bdcst_0, c_vec_5);
            c_vec_6 = _mm256_fmaddsub_pd(a_vec_0, bdcst_1, c_vec_6);
            c_vec_7 = _mm256_fmaddsub_pd(a_vec_1, bdcst_1, c_vec_7);

            // Scaling with beta, according to its type.
            switch( beta_mul_type )
            {
                case BLIS_MUL_ZERO :
                    break;

                case BLIS_MUL_ONE :
                    // Load C and add with the corresponding scratch register.
                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + 2));
                    c_vec_0 = _mm256_add_pd(c_vec_0, a_vec_0);
                    c_vec_1 = _mm256_add_pd(c_vec_1, a_vec_1);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + ldc + 2));
                    c_vec_2 = _mm256_add_pd(c_vec_2, a_vec_0);
                    c_vec_3 = _mm256_add_pd(c_vec_3, a_vec_1);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc*2));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + ldc*2 + 2));
                    c_vec_4 = _mm256_add_pd(c_vec_4, a_vec_0);
                    c_vec_5 = _mm256_add_pd(c_vec_5, a_vec_1);

                    a_vec_0 = _mm256_loadu_pd((double const*)(temp_cij + ldc*3));
                    a_vec_1 = _mm256_loadu_pd((double const*)(temp_cij + ldc*3 + 2));
                    c_vec_6 = _mm256_add_pd(c_vec_6, a_vec_0);
                    c_vec_7 = _mm256_add_pd(c_vec_7, a_vec_1);
                    break;

                default :
                    // Broadcast beta and redirect to the beta scaling macro.
                    bdcst_0 = _mm256_broadcast_sd((double const*)(&beta_real));
                    bdcst_1 = _mm256_broadcast_sd((double const*)(&beta_imag));

                    BETA_SCALING_C_MAIN(c_vec_0, c_vec_1, temp_cij);
                    BETA_SCALING_C_MAIN(c_vec_2, c_vec_3, temp_cij + ldc);
                    BETA_SCALING_C_MAIN(c_vec_4, c_vec_5, temp_cij + ldc*2);
                    BETA_SCALING_C_MAIN(c_vec_6, c_vec_7, temp_cij + ldc*3);

            }

            // Storing the result in C.
            _mm256_storeu_pd((double *)(temp_cij), c_vec_0);
            _mm256_storeu_pd((double *)(temp_cij + 2), c_vec_1);

            _mm256_storeu_pd((double *)(temp_cij + ldc), c_vec_2);
            _mm256_storeu_pd((double *)(temp_cij + ldc + 2), c_vec_3);

            _mm256_storeu_pd((double *)(temp_cij + ldc*2), c_vec_4);
            _mm256_storeu_pd((double *)(temp_cij + ldc*2 + 2), c_vec_5);

            _mm256_storeu_pd((double *)(temp_cij + ldc*3), c_vec_6);
            _mm256_storeu_pd((double *)(temp_cij + ldc*3 + 2), c_vec_7);

            // Adjusting the addresses of A and C for the next iteration.
            temp_cij += Z_MR;
            temp_ai += Z_MR;
        }

    }

}
