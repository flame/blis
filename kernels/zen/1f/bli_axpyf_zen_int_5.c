/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    __m128d xmm[2];
    double  d[4] __attribute__((aligned(64)));
} v4df_t;

typedef union
{
    __m128d v;
    double  d[2] __attribute__((aligned(64)));
} v2df_t;

/* Mask vector for vmaskmovps operations.
   For single precision, ymm register holds 8 floats.
   Mask uses -1 (all bits set) to load/store, 0 to skip. */
static const int32_t mask[8][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},           // 0 elements (not used)
    {-1, 0, 0, 0, 0, 0, 0, 0},          // 1 element
    {-1, -1, 0, 0, 0, 0, 0, 0},         // 2 elements
    {-1, -1, -1, 0, 0, 0, 0, 0},        // 3 elements
    {-1, -1, -1, -1, 0, 0, 0, 0},       // 4 elements
    {-1, -1, -1, -1, -1, 0, 0, 0},      // 5 elements
    {-1, -1, -1, -1, -1, -1, 0, 0},     // 6 elements
    {-1, -1, -1, -1, -1, -1, -1, 0}     // 7 elements
};

void bli_saxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float* restrict alpha,
       float* restrict a, inc_t inca, inc_t lda,
       float* restrict x, inc_t incx,
       float* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t     fuse_fac       = 5;

    dim_t           i;

    float* restrict a0;
    float* restrict a1;
    float* restrict a2;
    float* restrict a3;
    float* restrict a4;

    float* restrict y0;

    float           chi0, chi1, chi2, chi3, chi4;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_seq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        saxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_FLOAT, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            float* a1   = a + (0  )*inca + (i  )*lda;
            float* chi1 = x + (i  )*incx;
            float* y1   = y + (0  )*incy;
            float  alpha_chi1;

            bli_scopycjs( conjx, *chi1, alpha_chi1 );
            bli_sscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    a0   = a + 0*lda;
    a1   = a + 1*lda;
    a2   = a + 2*lda;
    a3   = a + 3*lda;
    a4   = a + 4*lda;
    y0   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );
    chi4 = *( x + 4*incx );

    // Scale each chi scalar by alpha.
    bli_sscals( *alpha, chi0 );
    bli_sscals( *alpha, chi1 );
    bli_sscals( *alpha, chi2 );
    bli_sscals( *alpha, chi3 );
    bli_sscals( *alpha, chi4 );

    // Typecast to 64 bit
    uint64_t m0 = (uint64_t)m;

    // If there are vectorized iterations, perform them with vector instructions.
    if ( inca == 1 && incy == 1 )
    {
        
        // Assembly Code
        begin_asm()

        /*
            rsi - > m
            rax - > a0
            rbx - > a1
            rcx - > a2
            rdx - > a3
            rdi - > a4
            r8  - > y0
            ymm0-ymm4 -> chi0-chi4 (broadcasted)
        */

        // Load pointers
        mov(var(a0), rax)
        mov(var(a1), rbx)
        mov(var(a2), rcx)
        mov(var(a3), rdx)
        mov(var(a4), rdi)
        mov(var(y0), r8)
        mov(var(m0), rsi)

        // Broadcast chi scalars to ymm registers
        vbroadcastss(var(chi0), ymm0)
        vbroadcastss(var(chi1), ymm1)
        vbroadcastss(var(chi2), ymm2)
        vbroadcastss(var(chi3), ymm3)
        vbroadcastss(var(chi4), ymm4)

        // ========================================================================================================================

        // Section to process blocks of 16 elements
        label(.BLOCK16)

        cmp(imm(16), rsi)                   // check if remaining elements >= 16
        jl(.BLOCK8)                         // else, goto block of size 8

        label(.LOOP16)

        // Load y values
        vmovups(mem(r8, 0*32), ymm5)        // ymm5 = y[0-7]
        vmovups(mem(r8, 1*32), ymm6)        // ymm6 = y[8-15]

        // Load a0 and perform FMA with chi0
        vmovups(mem(rax, 0*32), ymm7)       // ymm7 = a0[0-7]
        vmovups(mem(rax, 1*32), ymm8)       // ymm8 = a0[8-15]
        vfmadd231ps(ymm0, ymm7, ymm5)       // y[0-7] += chi0 * a0[0-7]
        vfmadd231ps(ymm0, ymm8, ymm6)       // y[8-15] += chi0 * a0[8-15]

        // Load a1 and perform FMA with chi1
        vmovups(mem(rbx, 0*32), ymm7)       // ymm7 = a1[0-7]
        vmovups(mem(rbx, 1*32), ymm8)       // ymm8 = a1[8-15]
        vfmadd231ps(ymm1, ymm7, ymm5)       // y[0-7] += chi1 * a1[0-7]
        vfmadd231ps(ymm1, ymm8, ymm6)       // y[8-15] += chi1 * a1[8-15]

        // Load a2 and perform FMA with chi2
        vmovups(mem(rcx, 0*32), ymm7)       // ymm7 = a2[0-7]
        vmovups(mem(rcx, 1*32), ymm8)       // ymm8 = a2[8-15]
        vfmadd231ps(ymm2, ymm7, ymm5)       // y[0-7] += chi2 * a2[0-7]
        vfmadd231ps(ymm2, ymm8, ymm6)       // y[8-15] += chi2 * a2[8-15]

        // Load a3 and perform FMA with chi3
        vmovups(mem(rdx, 0*32), ymm7)       // ymm7 = a3[0-7]
        vmovups(mem(rdx, 1*32), ymm8)       // ymm8 = a3[8-15]
        vfmadd231ps(ymm3, ymm7, ymm5)       // y[0-7] += chi3 * a3[0-7]
        vfmadd231ps(ymm3, ymm8, ymm6)       // y[8-15] += chi3 * a3[8-15]

        // Load a4 and perform FMA with chi4
        vmovups(mem(rdi, 0*32), ymm7)       // ymm7 = a4[0-7]
        vmovups(mem(rdi, 1*32), ymm8)       // ymm8 = a4[8-15]
        vfmadd231ps(ymm4, ymm7, ymm5)       // y[0-7] += chi4 * a4[0-7]
        vfmadd231ps(ymm4, ymm8, ymm6)       // y[8-15] += chi4 * a4[8-15]

        // Store y values
        vmovups(ymm5, mem(r8, 0*32))        // y[0-7] = ymm5
        vmovups(ymm6, mem(r8, 1*32))        // y[8-15] = ymm6

        // Increment pointers
        add(imm(16*4), rax)
        add(imm(16*4), rbx)
        add(imm(16*4), rcx)
        add(imm(16*4), rdx)
        add(imm(16*4), rdi)
        add(imm(16*4), r8)
        sub(imm(16), rsi)

        cmp(imm(15), rsi)
        jg(.LOOP16)

        // -----------------------------------------------------------

        // Section to process blocks of 8 elements
        label(.BLOCK8)

        cmp(imm(8), rsi)                    // check if remaining elements >= 8
        jl(.FRINGE)                         // else, goto fringe code

        // Load y values
        vmovups(mem(r8, 0*32), ymm5)        // ymm5 = y[0-7]

        // Load a0 and perform FMA with chi0
        vmovups(mem(rax, 0*32), ymm6)       // ymm6 = a0[0-7]
        vfmadd231ps(ymm0, ymm6, ymm5)       // y[0-7] += chi0 * a0[0-7]

        // Load a1 and perform FMA with chi1
        vmovups(mem(rbx, 0*32), ymm6)       // ymm6 = a1[0-7]
        vfmadd231ps(ymm1, ymm6, ymm5)       // y[0-7] += chi1 * a1[0-7]

        // Load a2 and perform FMA with chi2
        vmovups(mem(rcx, 0*32), ymm6)       // ymm6 = a2[0-7]
        vfmadd231ps(ymm2, ymm6, ymm5)       // y[0-7] += chi2 * a2[0-7]

        // Load a3 and perform FMA with chi3
        vmovups(mem(rdx, 0*32), ymm6)       // ymm6 = a3[0-7]
        vfmadd231ps(ymm3, ymm6, ymm5)       // y[0-7] += chi3 * a3[0-7]

        // Load a4 and perform FMA with chi4
        vmovups(mem(rdi, 0*32), ymm6)       // ymm6 = a4[0-7]
        vfmadd231ps(ymm4, ymm6, ymm5)       // y[0-7] += chi4 * a4[0-7]

        // Store y values
        vmovups(ymm5, mem(r8, 0*32))        // y[0-7] = ymm5

        // Increment pointers
        add(imm(8*4), rax)
        add(imm(8*4), rbx)
        add(imm(8*4), rcx)
        add(imm(8*4), rdx)
        add(imm(8*4), rdi)
        add(imm(8*4), r8)
        sub(imm(8), rsi)

        // -----------------------------------------------------------

        // Masked vector code for remaining elements (1-7)
        label(.FRINGE)

        cmp(imm(0), rsi)
        je(.END)

        // Load mask based on remaining element count
        lea(var(mask), r9)                  // r9 = &mask[0][0]
        lea(mem(, rsi, 8), r10)             // r10 = rsi * 8 (8 int32_t per row)
        lea(mem(, r10, 4), r10)             // r10 = r10 * 4 (4 bytes per int32_t) = rsi * 32
        add(r10, r9)                        // r9 = &mask[rsi][0]
        vmovdqu(mem(r9), ymm15)             // ymm15 = mask vector

        // Load y values with mask
        vmaskmovps(mem(r8), ymm15, ymm5)    // ymm5 = y[0-7] (masked)

        // Load a0 and perform FMA with chi0
        vmaskmovps(mem(rax), ymm15, ymm6)   // ymm6 = a0[0-7] (masked)
        vfmadd231ps(ymm0, ymm6, ymm5)       // y[0-7] += chi0 * a0[0-7]

        // Load a1 and perform FMA with chi1
        vmaskmovps(mem(rbx), ymm15, ymm6)   // ymm6 = a1[0-7] (masked)
        vfmadd231ps(ymm1, ymm6, ymm5)       // y[0-7] += chi1 * a1[0-7]

        // Load a2 and perform FMA with chi2
        vmaskmovps(mem(rcx), ymm15, ymm6)   // ymm6 = a2[0-7] (masked)
        vfmadd231ps(ymm2, ymm6, ymm5)       // y[0-7] += chi2 * a2[0-7]

        // Load a3 and perform FMA with chi3
        vmaskmovps(mem(rdx), ymm15, ymm6)   // ymm6 = a3[0-7] (masked)
        vfmadd231ps(ymm3, ymm6, ymm5)       // y[0-7] += chi3 * a3[0-7]

        // Load a4 and perform FMA with chi4
        vmaskmovps(mem(rdi), ymm15, ymm6)   // ymm6 = a4[0-7] (masked)
        vfmadd231ps(ymm4, ymm6, ymm5)       // y[0-7] += chi4 * a4[0-7]

        // Store y values with mask
        vmaskmovps(ymm5, ymm15, mem(r8))    // y[0-7] = ymm5 (masked)

        label(.END)

        end_asm
        (
            : // output operands
            : // input operands
              [m0]    "m"     (m0),
              [a0]    "m"     (a0),
              [a1]    "m"     (a1),
              [a2]    "m"     (a2),
              [a3]    "m"     (a3),
              [a4]    "m"     (a4),
              [y0]    "m"     (y0),
              [chi0]  "m"     (chi0),
              [chi1]  "m"     (chi1),
              [chi2]  "m"     (chi2),
              [chi3]  "m"     (chi3),
              [chi4]  "m"     (chi4),
              [mask]  "m"     (mask)
            : // register clobber list
              "ymm0",  "ymm1",  "ymm2",  "ymm3",
              "ymm4",  "ymm5",  "ymm6",  "ymm7",
              "ymm8",  "ymm15", "xmm0",  "xmm1",
              "xmm2",  "xmm3",  "xmm4",  "xmm5",
              "xmm6",  "rsi",   "rax",   "rbx",
              "rcx",   "rdx",   "rdi",   "r8",
              "r9",    "r10",   "memory"
        )
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *y0;

            const float a0c = *a0;
            const float a1c = *a1;
            const float a2c = *a2;
            const float a3c = *a3;
            const float a4c = *a4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *y0 = y0c;

            a0 += inca;
            a1 += inca;
            a2 += inca;
            a3 += inca;
            a4 += inca;
            y0 += incy;
        }

    }
}


// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t      fuse_fac       = 5;

    const dim_t      n_elem_per_reg = 4;

    dim_t            i;

    double* restrict av[5] __attribute__((aligned(64)));

    double* restrict y0;

    v4df_t           chiv[5], a_vec[20], yv[4];
    
    double           chi[5];

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        daxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            double* a1   = a + (0  )*inca + (i  )*lda;
            double* chi1 = x + (i  )*incx;
            double* y1   = y + (0  )*incy;
            double  alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.
    // av points to the 5 columns under consideration
    av[0]   = a + 0*lda;
    av[1]   = a + 1*lda;
    av[2]   = a + 2*lda;
    av[3]   = a + 3*lda;
    av[4]   = a + 4*lda;
    y0   = y;

    chi[0] = *( x + 0*incx );
    chi[1] = *( x + 1*incx );
    chi[2] = *( x + 2*incx );
    chi[3] = *( x + 3*incx );
    chi[4] = *( x + 4*incx );


    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi[0] );
    bli_dscals( *alpha, chi[1] );
    bli_dscals( *alpha, chi[2] );
    bli_dscals( *alpha, chi[3] );
    bli_dscals( *alpha, chi[4] );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
        chiv[0].v = _mm256_broadcast_sd( &chi[0] );
        chiv[1].v = _mm256_broadcast_sd( &chi[1] );
        chiv[2].v = _mm256_broadcast_sd( &chi[2] );
        chiv[3].v = _mm256_broadcast_sd( &chi[3] );
        chiv[4].v = _mm256_broadcast_sd( &chi[4] );

        // 16 elements of the result are computed per iteration
        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            yv[3].v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

            a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
            a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
            a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
            a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
            a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );

            a_vec[5].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
            a_vec[6].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
            a_vec[7].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
            a_vec[8].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
            a_vec[9].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );

            a_vec[10].v = _mm256_loadu_pd( av[0] + 2*n_elem_per_reg );
            a_vec[11].v = _mm256_loadu_pd( av[1] + 2*n_elem_per_reg );
            a_vec[12].v = _mm256_loadu_pd( av[2] + 2*n_elem_per_reg );
            a_vec[13].v = _mm256_loadu_pd( av[3] + 2*n_elem_per_reg );
            a_vec[14].v = _mm256_loadu_pd( av[4] + 2*n_elem_per_reg );

            a_vec[15].v = _mm256_loadu_pd( av[0] + 3*n_elem_per_reg );
            a_vec[16].v = _mm256_loadu_pd( av[1] + 3*n_elem_per_reg );
            a_vec[17].v = _mm256_loadu_pd( av[2] + 3*n_elem_per_reg );
            a_vec[18].v = _mm256_loadu_pd( av[3] + 3*n_elem_per_reg );
            a_vec[19].v = _mm256_loadu_pd( av[4] + 3*n_elem_per_reg );

            // perform : y += alpha * x;
            yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );

            yv[1].v = _mm256_fmadd_pd( a_vec[5].v, chiv[0].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[6].v, chiv[1].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[7].v, chiv[2].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[3].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[4].v, yv[1].v );

            yv[2].v = _mm256_fmadd_pd( a_vec[10].v, chiv[0].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[11].v, chiv[1].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[12].v, chiv[2].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[13].v, chiv[3].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[14].v, chiv[4].v, yv[2].v );

            yv[3].v = _mm256_fmadd_pd( a_vec[15].v, chiv[0].v, yv[3].v );
            yv[3].v = _mm256_fmadd_pd( a_vec[16].v, chiv[1].v, yv[3].v );
            yv[3].v = _mm256_fmadd_pd( a_vec[17].v, chiv[2].v, yv[3].v );
            yv[3].v = _mm256_fmadd_pd( a_vec[18].v, chiv[3].v, yv[3].v );
            yv[3].v = _mm256_fmadd_pd( a_vec[19].v, chiv[4].v, yv[3].v );

            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );
            _mm256_storeu_pd( (y0 + 2*n_elem_per_reg), yv[2].v );
            _mm256_storeu_pd( (y0 + 3*n_elem_per_reg), yv[3].v );

            y0 += n_elem_per_reg * 4;
            av[0] += n_elem_per_reg * 4;
            av[1] += n_elem_per_reg * 4;
            av[2] += n_elem_per_reg * 4;
            av[3] += n_elem_per_reg * 4;
            av[4] += n_elem_per_reg * 4;
        }

        // 12 elements of the result are computed per iteration
        for ( ; (i + 11) < m; i += 12 )
        {
            // Load the input values.
            yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

            a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
            a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
            a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
            a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
            a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );

            a_vec[5].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
            a_vec[6].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
            a_vec[7].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
            a_vec[8].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
            a_vec[9].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );

            a_vec[10].v = _mm256_loadu_pd( av[0] + 2*n_elem_per_reg );
            a_vec[11].v = _mm256_loadu_pd( av[1] + 2*n_elem_per_reg );
            a_vec[12].v = _mm256_loadu_pd( av[2] + 2*n_elem_per_reg );
            a_vec[13].v = _mm256_loadu_pd( av[3] + 2*n_elem_per_reg );
            a_vec[14].v = _mm256_loadu_pd( av[4] + 2*n_elem_per_reg );

            // perform : y += alpha * x;
            yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );

            yv[1].v = _mm256_fmadd_pd( a_vec[5].v, chiv[0].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[6].v, chiv[1].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[7].v, chiv[2].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[3].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[4].v, yv[1].v );

            yv[2].v = _mm256_fmadd_pd( a_vec[10].v, chiv[0].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[11].v, chiv[1].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[12].v, chiv[2].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[13].v, chiv[3].v, yv[2].v );
            yv[2].v = _mm256_fmadd_pd( a_vec[14].v, chiv[4].v, yv[2].v );

            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );
            _mm256_storeu_pd( (y0 + 2*n_elem_per_reg), yv[2].v );

            y0 += n_elem_per_reg * 3;
            av[0] += n_elem_per_reg * 3;
            av[1] += n_elem_per_reg * 3;
            av[2] += n_elem_per_reg * 3;
            av[3] += n_elem_per_reg * 3;
            av[4] += n_elem_per_reg * 3;
        }

        // 8 elements of the result are computed per iteration
        for (; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
            a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
            a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
            a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
            a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );

            a_vec[5].v = _mm256_loadu_pd( av[0] + 1*n_elem_per_reg );
            a_vec[6].v = _mm256_loadu_pd( av[1] + 1*n_elem_per_reg );
            a_vec[7].v = _mm256_loadu_pd( av[2] + 1*n_elem_per_reg );
            a_vec[8].v = _mm256_loadu_pd( av[3] + 1*n_elem_per_reg );
            a_vec[9].v = _mm256_loadu_pd( av[4] + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );

            yv[1].v = _mm256_fmadd_pd( a_vec[5].v, chiv[0].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[6].v, chiv[1].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[7].v, chiv[2].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[8].v, chiv[3].v, yv[1].v );
            yv[1].v = _mm256_fmadd_pd( a_vec[9].v, chiv[4].v, yv[1].v );

            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), yv[1].v );

            y0 += n_elem_per_reg * 2;
            av[0] += n_elem_per_reg * 2;
            av[1] += n_elem_per_reg * 2;
            av[2] += n_elem_per_reg * 2;
            av[3] += n_elem_per_reg * 2;
            av[4] += n_elem_per_reg * 2;
        }

        // 4 elements of the result are computed per iteration
        for( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            a_vec[0].v = _mm256_loadu_pd( av[0] + 0*n_elem_per_reg );
            a_vec[1].v = _mm256_loadu_pd( av[1] + 0*n_elem_per_reg );
            a_vec[2].v = _mm256_loadu_pd( av[2] + 0*n_elem_per_reg );
            a_vec[3].v = _mm256_loadu_pd( av[3] + 0*n_elem_per_reg );
            a_vec[4].v = _mm256_loadu_pd( av[4] + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            yv[0].v = _mm256_fmadd_pd( a_vec[0].v, chiv[0].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[1].v, chiv[1].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[2].v, chiv[2].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[3].v, chiv[3].v, yv[0].v );
            yv[0].v = _mm256_fmadd_pd( a_vec[4].v, chiv[4].v, yv[0].v );

            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), yv[0].v );

            y0 += n_elem_per_reg;
            av[0] += n_elem_per_reg;
            av[1] += n_elem_per_reg;
            av[2] += n_elem_per_reg;
            av[3] += n_elem_per_reg;
            av[4] += n_elem_per_reg;
        }
        
        // Remaining elements are handled using masked operations
        dim_t m_rem = m - i;
        if( m_rem > 0 )
        {
            // Masked operations can handle upto 4 doubles at a time
            __m256i indices = _mm256_setr_epi64x( 0, 1, 2, 3 );   // indices <- [0, 1, 2, 3] 

            // _mm256_set1_epi64x: broadcasts m_rem to all lanes of a 256 bit register, m_rem_YMM[:] <- m_rem
            // _mm256_cmpgt_epi64: m_rem_YMM[i] > indices[i] ? -1 : 0
            // This operation sets -1(0xFF..F) to all the lanes of the mask where m_rem > indices[i]
            // For example, if m_rem == 2, mask <- [-1, -1, 0, 0], this mask is subsequently
            // used to load 2 64 bit elements from the corresponding memory locations
            __m256i mask = _mm256_cmpgt_epi64( _mm256_set1_epi64x( m_rem ), indices ); // mask[i] <- m_rem > indices[i] ? -1 : 0
            
            // Perform the masked loads
            __m256d y0cv = _mm256_maskload_pd( y0, mask );
            __m256d a0cv = _mm256_maskload_pd( av[0], mask );
            __m256d a1cv = _mm256_maskload_pd( av[1], mask );
            __m256d a2cv = _mm256_maskload_pd( av[2], mask );
            __m256d a3cv = _mm256_maskload_pd( av[3], mask );
            __m256d a4cv = _mm256_maskload_pd( av[4], mask );
            
            // Perform FMAs on the masked registers
            y0cv = _mm256_fmadd_pd( chiv[0].v, a0cv, y0cv );
            y0cv = _mm256_fmadd_pd( chiv[1].v, a1cv, y0cv );
            y0cv = _mm256_fmadd_pd( chiv[2].v, a2cv, y0cv );
            y0cv = _mm256_fmadd_pd( chiv[3].v, a3cv, y0cv );
            y0cv = _mm256_fmadd_pd( chiv[4].v, a4cv, y0cv );
            
            // Masked store to y0
            _mm256_maskstore_pd( y0, mask, y0cv );
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur later,
        // especially if BLIS is compiled with -mfpmath=sse).
        _mm256_zeroupper();
    }
    else
    {
        /*
        * DAXPYF implementation for non-unit strides (inca/incy != 1).
        * While the loop remains scalar due to non-contiguous memory access, we use
        * SSE scalar intrinsics (_mm_fmadd_sd) to leverage the hardware's Fused
        * Multiply-Add (FMA) unit. This maintains higher precision by 
        * performing the multiply-add in a single step.
        */
        __m128d chi0v_s = _mm_load_sd( &chi[0] ); // chi0v_s[0] <- chi[0]
                                                  // chi0v_s[1] <- 0.0 (Unused)
        __m128d chi1v_s = _mm_load_sd( &chi[1] ); // chi1v_s[0] <- chi[1]
                                                  // chi1v_s[1] <- 0.0 (Unused)
        __m128d chi2v_s = _mm_load_sd( &chi[2] ); // chi2v_s[0] <- chi[2]
                                                  // chi2v_s[1] <- 0.0 (Unused)
        __m128d chi3v_s = _mm_load_sd( &chi[3] ); // chi3v_s[0] <- chi[3]
                                                  // chi3v_s[1] <- 0.0 (Unused)
        __m128d chi4v_s = _mm_load_sd( &chi[4] ); // chi4v_s[0] <- chi[4]
                                                  // chi4v_s[1] <- 0.0 (Unused)

        for ( i = 0; (i + 0) < m ; ++i )
        {
            // Load single double into low lane of __m128d
            __m128d y0cv_s = _mm_load_sd( y0 );     // y0cv_s[0] <- y0[0]
                                                    // y0cv_s[1] <- 0.0 (Unused)
            __m128d a0cv_s = _mm_load_sd( av[0] );  // a0cv_s[0] <- av[0]
                                                    // a0cv_s[1] <- 0.0 (Unused)
            __m128d a1cv_s = _mm_load_sd( av[1] );  // a1cv_s[0] <- av[1]
                                                    // a1cv_s[1] <- 0.0 (Unused)
            __m128d a2cv_s = _mm_load_sd( av[2] );  // a2cv_s[0] <- av[2]
                                                    // a2cv_s[1] <- 0.0 (Unused)
            __m128d a3cv_s = _mm_load_sd( av[3] );  // a3cv_s[0] <- av[3]
                                                    // a3cv_s[1] <- 0.0 (Unused)
            __m128d a4cv_s = _mm_load_sd( av[4] );  // a4cv_s[0] <- av[4]
                                                    // a4cv_s[1] <- 0.0 (Unused)
            
            // _mm_fmadd_sd performs a scalar fused multiply-add (FMA) 
            // on the low 64-bit double-precision element of XMM registers.
            // This operation is done in one fused instruction
            // with one rounding (compared to two in the scalar case)
            y0cv_s = _mm_fmadd_sd( chi0v_s, a0cv_s, y0cv_s ); // y0cv_s[0] <- (chi0v_s[0] * a0cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi1v_s, a1cv_s, y0cv_s ); // y0cv_s[0] <- (chi1v_s[0] * a1cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi1v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi2v_s, a2cv_s, y0cv_s ); // y0cv_s[0] <- (chi2v_s[0] * a2cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi2v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi3v_s, a3cv_s, y0cv_s ); // y0cv_s[0] <- (chi3v_s[0] * a3cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi3v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi4v_s, a4cv_s, y0cv_s ); // y0cv_s[0] <- (chi4v_s[0] * a4cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi4v_s[1] (Unused)

            // store the accumulated value (lower lane) 
            _mm_store_sd( y0, y0cv_s ); // y0 <- y0cv_s[0]

            av[0] += inca;
            av[1] += inca;
            av[2] += inca;
            av[3] += inca;
            av[4] += inca;
            y0 += incy;
        }
    }
}

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_16x2
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t      fuse_fac       = 2;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 4;

    dim_t            i;

    double* restrict a0;
    double* restrict a1;

    double* restrict y0;

    v4df_t           chi0v, chi1v;

    v4df_t           a00v, a01v;

    v4df_t           a10v, a11v;

    v4df_t           a20v, a21v;

    v4df_t           a30v, a31v;

    v4df_t           y0v, y1v, y2v, y3v;

    double           chi0, chi1;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        daxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            double* a1   = a + (0  )*inca + (i  )*lda;
            double* chi1 = x + (i  )*incx;
            double* y1   = y + (0  )*incy;
            double  alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    a0   = a + 0*lda;
    a1   = a + 1*lda;

    y0   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );


    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi0 );
    bli_dscals( *alpha, chi1 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
        chi0v.v = _mm256_broadcast_sd( &chi0 );
        chi1v.v = _mm256_broadcast_sd( &chi1 );

        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            y3v.v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( a0 + 2*n_elem_per_reg );
            a30v.v = _mm256_loadu_pd( a0 + 3*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( a1 + 2*n_elem_per_reg );
            a31v.v = _mm256_loadu_pd( a1 + 3*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a30v.v, chi0v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a31v.v, chi1v.v, y3v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(y0 + 2*n_elem_per_reg), y2v.v );
            _mm256_storeu_pd( (double *)(y0 + 3*n_elem_per_reg), y3v.v );

            y0 += n_iter_unroll * n_elem_per_reg;
            a0 += n_iter_unroll * n_elem_per_reg;
            a1 += n_iter_unroll * n_elem_per_reg;
        }

        for ( ; (i + 11) < m; i += 12 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( a0 + 2*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( a1 + 2*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(y0 + 2*n_elem_per_reg), y2v.v );

            y0 += 3 * n_elem_per_reg;
            a0 += 3 * n_elem_per_reg;
            a1 += 3 * n_elem_per_reg;
        }
        for ( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );

            y0 += 2 * n_elem_per_reg;
            a0 += 2 * n_elem_per_reg;
            a1 += 2 * n_elem_per_reg;
        }

        for ( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg;
            a0 += n_elem_per_reg;
            a1 += n_elem_per_reg;
        }

        // Handle the remaining cases using masked operations
        dim_t m_rem = m - i;
        if( m_rem > 0 )
        {
            // Masked operations can handle upto 4 doubles at a time
            __m256i indices = _mm256_setr_epi64x( 0, 1, 2, 3 );  // indices <- [0, 1, 2, 3] 

            // _mm256_set1_epi64x: broadcasts m_rem to all lanes of a 256 bit register, m_rem_YMM[:] <- m_rem
            // _mm256_cmpgt_epi64: m_rem_YMM[i] > indices[i] ? -1 : 0
            // This operation sets -1(0xFF..F) to all the lanes of the mask where m_rem > indices[i]
            // For example, if m_rem == 2, mask <- [-1, -1, 0, 0], this mask is subsequently
            // used to load 2 64 bit elements from the corresponding memory locations
            __m256i mask = _mm256_cmpgt_epi64(_mm256_set1_epi64x( m_rem ), indices ); // mask[i] <- m_rem > indices[i] ? -1 : 0

            // Perform the masked loads
            y0v.v = _mm256_maskload_pd( y0, mask );
            a00v.v = _mm256_maskload_pd( a0, mask );
            a01v.v = _mm256_maskload_pd( a1, mask );

            // Perform the masked FMA on the masked registers
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );

            // Masked store to y0
            _mm256_maskstore_pd( y0, mask, y0v.v );
        }
        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur later,
        // especially if BLIS is compiled with -mfpmath=sse).
        _mm256_zeroupper();
    }
    else
    {
        /*
        * DAXPYF implementation for non-unit strides (inca/incy != 1).
        * While the loop remains scalar due to non-contiguous memory access, we use
        * SSE scalar intrinsics (_mm_fmadd_sd) to leverage the hardware's Fused
        * Multiply-Add (FMA) unit. This maintains higher precision by 
        * performing the multiply-add in a single step.
        */

        __m128d chi0v_s = _mm_load_sd( &chi0 ); // chi0v_s[0] <- chi0
                                                // chi0v_s[1] <- 0.0 (Unused)
        __m128d chi1v_s = _mm_load_sd( &chi1 ); // chi1v_s[0] <- chi1
                                                // chi1v_s[1] <- 0.0 (Unused)
        
        for ( i = 0; (i + 0) < m ; ++i )
        {
            // Load single double into low lane of __m128d
            __m128d y0cv_s = _mm_load_sd( y0 );  // y0cv_s[0] <- y0[0]
                                                 // y0cv_s[1] <- 0.0 (Unused)
            __m128d a0cv_s = _mm_load_sd( a0 );  // a0cv_s[0] <- a0[0]
                                                 // a0cv_s[1] <- 0.0 (Unused)
            __m128d a1cv_s = _mm_load_sd( a1 );  // a1cv_s[0] <- a1[0]
                                                 // a1cv_s[1] <- 0.0 (Unused)

            // _mm_fmadd_sd performs a scalar fused multiply-add (FMA) 
            // on the low 64-bit double-precision element of XMM registers.
            // This operation is done in one fused instruction
            // with one rounding (compared to two in the scalar case)
            y0cv_s = _mm_fmadd_sd( chi0v_s, a0cv_s, y0cv_s ); // y0cv_s[0] <- (chi0v_s[0] * a0cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi1v_s, a1cv_s, y0cv_s ); // y0cv_s[0] <- (chi1v_s[0] * a1cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi1v_s[1] (Unused)
            
            // store the accumulated value (lower lane) 
            _mm_store_sd( y0, y0cv_s ); // y0 <- y0cv_s[0]

            a0 += inca;
            a1 += inca;
            y0 += incy;
        }
    }
}

// -----------------------------------------------------------------------------
void bli_daxpyf_zen_int_16x4
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t      fuse_fac       = 4;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 4;

    dim_t            i;

    double* restrict a0;
    double* restrict a1;
    double* restrict a2;
    double* restrict a3;

    double* restrict y0;

    v4df_t           chi0v, chi1v, chi2v, chi3v;

    v4df_t           a00v, a01v, a02v, a03v;

    v4df_t           a10v, a11v, a12v, a13v;

    v4df_t           a20v, a21v, a22v, a23v;

    v4df_t           a30v, a31v, a32v, a33v;

    v4df_t           y0v, y1v, y2v, y3v;

    double           chi0, chi1, chi2, chi3;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        daxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            double* a1   = a + (0  )*inca + (i  )*lda;
            double* chi1 = x + (i  )*incx;
            double* y1   = y + (0  )*incy;
            double  alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    a0   = a + 0*lda;
    a1   = a + 1*lda;
    a2   = a + 2*lda;
    a3   = a + 3*lda;

    y0   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );

    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi0 );
    bli_dscals( *alpha, chi1 );
    bli_dscals( *alpha, chi2 );
    bli_dscals( *alpha, chi3 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
        chi0v.v = _mm256_broadcast_sd( &chi0 );
        chi1v.v = _mm256_broadcast_sd( &chi1 );
        chi2v.v = _mm256_broadcast_sd( &chi2 );
        chi3v.v = _mm256_broadcast_sd( &chi3 );

        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
            y3v.v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( a0 + 2*n_elem_per_reg );
            a30v.v = _mm256_loadu_pd( a0 + 3*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( a1 + 2*n_elem_per_reg );
            a31v.v = _mm256_loadu_pd( a1 + 3*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( a2 + 1*n_elem_per_reg );
            a22v.v = _mm256_loadu_pd( a2 + 2*n_elem_per_reg );
            a32v.v = _mm256_loadu_pd( a2 + 3*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( a3 + 1*n_elem_per_reg );
            a23v.v = _mm256_loadu_pd( a3 + 2*n_elem_per_reg );
            a33v.v = _mm256_loadu_pd( a3 + 3*n_elem_per_reg );

        // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a30v.v, chi0v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a31v.v, chi1v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a22v.v, chi2v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a32v.v, chi2v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a23v.v, chi3v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a33v.v, chi3v.v, y3v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(y0 + 2*n_elem_per_reg), y2v.v );
            _mm256_storeu_pd( (double *)(y0 + 3*n_elem_per_reg), y3v.v );

            y0 += n_iter_unroll * n_elem_per_reg;
            a0 += n_iter_unroll * n_elem_per_reg;
            a1 += n_iter_unroll * n_elem_per_reg;
            a2 += n_iter_unroll * n_elem_per_reg;
            a3 += n_iter_unroll * n_elem_per_reg;
        }

        for ( ; (i + 11) < m; i += 12 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( a0 + 2*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( a1 + 2*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( a2 + 1*n_elem_per_reg );
            a22v.v = _mm256_loadu_pd( a2 + 2*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( a3 + 1*n_elem_per_reg );
            a23v.v = _mm256_loadu_pd( a3 + 2*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a22v.v, chi2v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a23v.v, chi3v.v, y2v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(y0 + 2*n_elem_per_reg), y2v.v );

            y0 += 3 * n_elem_per_reg;
            a0 += 3 * n_elem_per_reg;
            a1 += 3 * n_elem_per_reg;
            a2 += 3 * n_elem_per_reg;
            a3 += 3 * n_elem_per_reg;
        }

        for ( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( a2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( a3 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );

            y0 += 2 * n_elem_per_reg;
            a0 += 2 * n_elem_per_reg;
            a1 += 2 * n_elem_per_reg;
            a2 += 2 * n_elem_per_reg;
            a3 += 2 * n_elem_per_reg;
        }


        for ( ; (i + 3) < m; i += 4)
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg;
            a0 += n_elem_per_reg;
            a1 += n_elem_per_reg;
            a2 += n_elem_per_reg;
            a3 += n_elem_per_reg;
        }

        // Handle the remaining cases using masked operations
        dim_t m_rem = m - i;
        if ( m_rem > 0 )
        {
            // Masked operations can handle upto 4 doubles at a time
            __m256i indices = _mm256_setr_epi64x( 0, 1, 2, 3 );   // indices <- [0, 1, 2, 3] 

            // _mm256_set1_epi64x: broadcasts m_rem to all lanes of a 256 bit register, m_rem_YMM[:] <- m_rem
            // _mm256_cmpgt_epi64: m_rem_YMM[i] > indices[i] ? -1 : 0
            // This operation sets -1(0xFF..F) to all the lanes of the mask where m_rem > indices[i]
            // For example, if m_rem == 2, mask <- [-1, -1, 0, 0], this mask is subsequently
            // used to load 2 64 bit elements from the corresponding memory locations
            __m256i mask = _mm256_cmpgt_epi64( _mm256_set1_epi64x( m_rem ), indices ); // mask[i] <- m_rem > indices[i] ? -1 : 0

            // Perform the masked loads
            y0v.v  = _mm256_maskload_pd( y0, mask );
            a00v.v = _mm256_maskload_pd( a0, mask );
            a01v.v = _mm256_maskload_pd( a1, mask );
            a02v.v = _mm256_maskload_pd( a2, mask );
            a03v.v = _mm256_maskload_pd( a3, mask );

            // Perform the masked FMA on the masked registers
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );

            // Masked store to y0
            _mm256_maskstore_pd( y0, mask, y0v.v );
        }
        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur later,
        // especially if BLIS is compiled with -mfpmath=sse).
        _mm256_zeroupper();
    }
    else
    {
        /*
        * DAXPYF implementation for non-unit strides (inca/incy != 1).
        * While the loop remains scalar due to non-contiguous memory access, we use
        * SSE scalar intrinsics (_mm_fmadd_sd) to leverage the hardware's Fused
        * Multiply-Add (FMA) unit. This maintains higher precision by 
        * performing the multiply-add in a single step.
        */

        __m128d chi0v_s = _mm_load_sd( &chi0 ); // chi0v_s[0] <- chi0[0]
                                                // chi0v_s[1] <- 0.0 (Unused)
        __m128d chi1v_s = _mm_load_sd( &chi1 ); // chi1v_s[0] <- chi1[0]
                                                // chi1v_s[1] <- 0.0 (Unused)
        __m128d chi2v_s = _mm_load_sd( &chi2 ); // chi2v_s[0] <- chi2[0]
                                                // chi2v_s[1] <- 0.0 (Unused)
        __m128d chi3v_s = _mm_load_sd( &chi3 ); // chi3v_s[0] <- chi3[0]
                                                // chi3v_s[1] <- 0.0 (Unused)

        for ( i = 0; (i + 0) < m ; ++i )
        {
            // Load single double into low lane of __m128d
            __m128d y0cv_s = _mm_load_sd( y0 ); // y0cv_s[0] <- y0[0]
                                                // y0cv_s[1] <- 0.0 (Unused)
            __m128d a0cv_s = _mm_load_sd( a0 ); // a0cv_s[0] <- a0[0]
                                                // a0cv_s[1] <- 0.0 (Unused)
            __m128d a1cv_s = _mm_load_sd( a1 ); // a1cv_s[0] <- a1[0]
                                                // a1cv_s[1] <- 0.0 (Unused)
            __m128d a2cv_s = _mm_load_sd( a2 ); // a2cv_s[0] <- a2[0]
                                                // a2cv_s[1] <- 0.0 (Unused)
            __m128d a3cv_s = _mm_load_sd( a3 ); // a3cv_s[0] <- a3[0]
                                                // a3cv_s[1] <- 0.0 (Unused)

            // _mm_fmadd_sd performs a scalar fused multiply-add (FMA) 
            // on the low 64-bit double-precision element of XMM registers.
            // This operation is done in one fused instruction
            // with one rounding (compared to two in the scalar case)
            y0cv_s = _mm_fmadd_sd( chi0v_s, a0cv_s, y0cv_s ); // y0cv_s[0] <- (chi0v_s[0] * a0cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi1v_s, a1cv_s, y0cv_s ); // y0cv_s[0] <- (chi1v_s[0] * a1cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi2v_s, a2cv_s, y0cv_s ); // y0cv_s[0] <- (chi2v_s[0] * a2cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)
            y0cv_s = _mm_fmadd_sd( chi3v_s, a3cv_s, y0cv_s ); // y0cv_s[0] <- (chi3v_s[0] * a3cv_s[0]) + y0cv_s[0]
                                                              // y0cv_s[1] <- chi0v_s[1] (Unused)

            // store the accumulated value (lower lane) 
            _mm_store_sd( y0, y0cv_s ); // y0[0] <- y0cv_s[0]

            a0 += inca;
            a1 += inca;
            a2 += inca;
            a3 += inca;

            y0 += incy;
        }
    }
}

// -----------------------------------------------------------------------------

void bli_caxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       scomplex* restrict alpha,
       scomplex* restrict a, inc_t inca, inc_t lda,
       scomplex* restrict x, inc_t incx,
       scomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t         fuse_fac       = 5;

    const dim_t         n_elem_per_reg = 4;

    dim_t               i = 0;
    dim_t               setPlusOne = 1;

    v8sf_t              chi0v, chi1v, chi2v, chi3v, chi4v;
    v8sf_t              chi5v, chi6v, chi7v, chi8v, chi9v;

    v8sf_t              a00v, a01v, a02v, a03v, a04v;
    v8sf_t              a05v, a06v, a07v, a08v, a09v;
#if 0
    v8sf_t              a10v, a11v, a12v, a13v, a14v;
    v8sf_t              a15v, a16v, a17v, a18v, a19v;
    v8sf_t              y1v;
#endif
    v8sf_t              y0v;
    v8sf_t              setMinus, setPlus;

    scomplex* restrict  a0;
    scomplex* restrict  a1;
    scomplex* restrict  a2;
    scomplex* restrict  a3;
    scomplex* restrict  a4;

    scomplex* restrict  y0;

    scomplex            chi0;
    scomplex            chi1;
    scomplex            chi2;
    scomplex            chi3;
    scomplex            chi4;

    if ( bli_is_conj(conja) ){
        setPlusOne = -1;
    }

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_ceq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        caxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            scomplex* a1   = a + (0  )*inca + (i  )*lda;
            scomplex* chi1 = x + (i  )*incx;
            scomplex* y1   = y + (0  )*incy;
            scomplex  alpha_chi1;

            bli_ccopycjs( conjx, *chi1, alpha_chi1 );
            bli_cscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }


    // At this point, we know that b_n is exactly equal to the fusing factor.

    a0   = a + 0*lda;
    a1   = a + 1*lda;
    a2   = a + 2*lda;
    a3   = a + 3*lda;
    a4   = a + 4*lda;
    y0   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );
    chi4 = *( x + 4*incx );

    scomplex *pchi0 = x + 0*incx ;
    scomplex *pchi1 = x + 1*incx ;
    scomplex *pchi2 = x + 2*incx ;
    scomplex *pchi3 = x + 3*incx ;
    scomplex *pchi4 = x + 4*incx ;

    bli_ccopycjs( conjx, *pchi0, chi0 );
    bli_ccopycjs( conjx, *pchi1, chi1 );
    bli_ccopycjs( conjx, *pchi2, chi2 );
    bli_ccopycjs( conjx, *pchi3, chi3 );
    bli_ccopycjs( conjx, *pchi4, chi4 );

    // Scale each chi scalar by alpha.
    bli_cscals( *alpha, chi0 );
    bli_cscals( *alpha, chi1 );
    bli_cscals( *alpha, chi2 );
    bli_cscals( *alpha, chi3 );
    bli_cscals( *alpha, chi4 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_ss( &chi0.real );
    chi1v.v = _mm256_broadcast_ss( &chi1.real );
    chi2v.v = _mm256_broadcast_ss( &chi2.real );
    chi3v.v = _mm256_broadcast_ss( &chi3.real );
    chi4v.v = _mm256_broadcast_ss( &chi4.real );

    chi5v.v = _mm256_broadcast_ss( &chi0.imag );
    chi6v.v = _mm256_broadcast_ss( &chi1.imag );
    chi7v.v = _mm256_broadcast_ss( &chi2.imag );
    chi8v.v = _mm256_broadcast_ss( &chi3.imag );
    chi9v.v = _mm256_broadcast_ss( &chi4.imag );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        setMinus.v = _mm256_set_ps( -1, 1, -1, 1, -1, 1, -1, 1 );

        setPlus.v = _mm256_set1_ps( 1 );
        if ( bli_is_conj(conja) ){
            setPlus.v = _mm256_set_ps( -1, 1, -1, 1, -1, 1, -1, 1 );
        }

        /*
         y := y + alpha * conja(A) * conjx(x)

          nn
          (ar + ai) (xr + xi)
          ar * xr - ai * xi
          ar * xi + ai * xr

         cc : (ar - ai) (xr - xi)
          ar * xr - ai * xi
          -(ar * xi + ai * xr)

          nc : (ar + ai) (xr - xi)
           ar * xr + ai * xi
          -(ar * xi - ai * xr)

          cn : (ar - ai) (xr + xi)
           ar * xr + ai * xi
          ar * xi - ai * xr

        */

        i = 0;
#if 0 //Low performance
        for( i = 0; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( (float*) (y0 + 0*n_elem_per_reg ));
            y1v.v = _mm256_loadu_ps( (float*) (y0 + 1*n_elem_per_reg ));

            a00v.v = _mm256_loadu_ps( (float*) (a0 + 0*n_elem_per_reg ));
            a10v.v = _mm256_loadu_ps( (float*) (a0 + 1*n_elem_per_reg ));

            a01v.v = _mm256_loadu_ps( (float*) (a1 + 0*n_elem_per_reg ));
            a11v.v = _mm256_loadu_ps( (float*) (a1 + 1*n_elem_per_reg ));

            a02v.v = _mm256_loadu_ps( (float*) (a2 + 0*n_elem_per_reg ));
            a12v.v = _mm256_loadu_ps( (float*) (a2 + 1*n_elem_per_reg ));

            a03v.v = _mm256_loadu_ps( (float*) (a3 + 0*n_elem_per_reg ));
            a13v.v = _mm256_loadu_ps( (float*) (a3 + 1*n_elem_per_reg ));

            a04v.v = _mm256_loadu_ps( (float*) (a4 + 0*n_elem_per_reg ));
            a14v.v = _mm256_loadu_ps( (float*) (a4 + 1*n_elem_per_reg ));

            a00v.v = _mm256_mul_ps( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_ps( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_ps( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_ps( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_ps( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_ps( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_ps( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_ps( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_ps( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_ps( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_ps( a05v.v, 0xB1 );
            a06v.v = _mm256_permute_ps( a06v.v, 0xB1 );
            a07v.v = _mm256_permute_ps( a07v.v, 0xB1 );
            a08v.v = _mm256_permute_ps( a08v.v, 0xB1 );
            a09v.v = _mm256_permute_ps( a09v.v, 0xB1 );

            a10v.v = _mm256_mul_ps( a10v.v, setPlus.v );
            a11v.v = _mm256_mul_ps( a11v.v, setPlus.v );
            a12v.v = _mm256_mul_ps( a12v.v, setPlus.v );
            a13v.v = _mm256_mul_ps( a13v.v, setPlus.v );
            a14v.v = _mm256_mul_ps( a14v.v, setPlus.v );

            a15v.v = _mm256_mul_ps( a10v.v, setMinus.v );
            a16v.v = _mm256_mul_ps( a11v.v, setMinus.v );
            a17v.v = _mm256_mul_ps( a12v.v, setMinus.v );
            a18v.v = _mm256_mul_ps( a13v.v, setMinus.v );
            a19v.v = _mm256_mul_ps( a14v.v, setMinus.v );

            a15v.v = _mm256_permute_ps( a15v.v, 0xB1 );
            a16v.v = _mm256_permute_ps( a16v.v, 0xB1 );
            a17v.v = _mm256_permute_ps( a17v.v, 0xB1 );
            a18v.v = _mm256_permute_ps( a18v.v, 0xB1 );
            a19v.v = _mm256_permute_ps( a19v.v, 0xB1 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_ps( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a09v.v, chi9v.v, y0v.v );

            // For next 4 elements perform : y += alpha * x;
            y1v.v = _mm256_fmadd_ps( a10v.v, chi0v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a11v.v, chi1v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a12v.v, chi2v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a13v.v, chi3v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a14v.v, chi4v.v, y1v.v );

            y1v.v = _mm256_fmadd_ps( a15v.v, chi5v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a16v.v, chi6v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a17v.v, chi7v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a18v.v, chi8v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a19v.v, chi9v.v, y1v.v );

            // Store the output.
            _mm256_storeu_ps( (float *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_ps( (float *)(y0 + 1*n_elem_per_reg), y1v.v );

            y0 += n_elem_per_reg * n_iter_unroll;
            a0 += n_elem_per_reg * n_iter_unroll;
            a1 += n_elem_per_reg * n_iter_unroll;
            a2 += n_elem_per_reg * n_iter_unroll;
            a3 += n_elem_per_reg * n_iter_unroll;
            a4 += n_elem_per_reg * n_iter_unroll;
        }
#endif
        for( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( (float*) (y0 + 0*n_elem_per_reg ));

            a00v.v = _mm256_loadu_ps( (float*) (a0 + 0*n_elem_per_reg ));
            a01v.v = _mm256_loadu_ps( (float*) (a1 + 0*n_elem_per_reg ));
            a02v.v = _mm256_loadu_ps( (float*) (a2 + 0*n_elem_per_reg ));
            a03v.v = _mm256_loadu_ps( (float*) (a3 + 0*n_elem_per_reg ));
            a04v.v = _mm256_loadu_ps( (float*) (a4 + 0*n_elem_per_reg ));

            a00v.v = _mm256_mul_ps( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_ps( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_ps( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_ps( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_ps( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_ps( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_ps( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_ps( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_ps( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_ps( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_ps( a05v.v, 0xB1 );
            a06v.v = _mm256_permute_ps( a06v.v, 0xB1 );
            a07v.v = _mm256_permute_ps( a07v.v, 0xB1 );
            a08v.v = _mm256_permute_ps( a08v.v, 0xB1 );
            a09v.v = _mm256_permute_ps( a09v.v, 0xB1 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_ps( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a09v.v, chi9v.v, y0v.v );

            // Store the output.
            _mm256_storeu_ps( (float *)(y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg ;
            a0 += n_elem_per_reg ;
            a1 += n_elem_per_reg ;
            a2 += n_elem_per_reg ;
            a3 += n_elem_per_reg ;
            a4 += n_elem_per_reg ;
        }

        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *y0;

            const scomplex a0c = *a0;
            const scomplex a1c = *a1;
            const scomplex a2c = *a2;
            const scomplex a3c = *a3;
            const scomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

            *y0 = y0c;

            a0 += 1;
            a1 += 1;
            a2 += 1;
            a3 += 1;
            a4 += 1;
            y0 += 1;
        }

    }
    else
    {
        for ( ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *y0;
            const scomplex a0c = *a0;
            const scomplex a1c = *a1;
            const scomplex a2c = *a2;
            const scomplex a3c = *a3;
            const scomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

            *y0 = y0c;

            a0 += inca;
            a1 += inca;
            a2 += inca;
            a3 += inca;
            a4 += inca;
            y0 += incy;
        }
    }
}


//------------------------------------------------------------------------------
/**
 * Following kernel performs axpyf operation on dcomplex data.
 * Operate over 5 columns of a matrix at a time and march through
 * rows in steps of 4 or 2.
 * For optimal performance, it separate outs imaginary and real
 * components of chis and broadcast them into separate ymm vector
 * registers.
 * By doing so it avoids necessity of permute operation to get the
 * final result of dcomp-lex multiplication.
 */
void bli_zaxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
	const dim_t              fuse_fac       = 5;

	const dim_t              n_elem_per_reg = 2;
	const dim_t              n_iter_unroll  = 2;

	dim_t                    i = 0;
	dim_t                    setPlusOne = 1;

	v4df_t                   chi0v, chi1v, chi2v, chi3v, chi4v;
	v4df_t                   chi5v, chi6v, chi7v, chi8v, chi9v;

	v4df_t                   a00v, a01v, a02v, a03v, a04v;

	v4df_t                   a10v, a11v, a12v, a13v, a14v;

	v4df_t                   y0v, y1v, y2v, y3v;
	v4df_t                   r0v, r1v, conjv;

	dcomplex                 chi0, chi1, chi2, chi3, chi4;
	dcomplex* restrict       a0;
	dcomplex* restrict       a1;
	dcomplex* restrict       a2;
	dcomplex* restrict       a3;
	dcomplex* restrict       a4;

	dcomplex*                restrict y0;


	if ( bli_is_conj(conja) ){
		setPlusOne = -1;
	}

	// If either dimension is zero, or if alpha is zero, return early.
	if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

	// If b_n is not equal to the fusing factor, then perform the entire
	// operation as a loop over axpyv.
	if ( b_n != fuse_fac )
	{
		zaxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_AXPYV_KER, cntx );

		for ( i = 0; i < b_n; ++i )
		{
			dcomplex* a1   = a + (0  )*inca + (i  )*lda;
			dcomplex* chi1 = x + (i  )*incx;
			dcomplex* y1   = y + (0  )*incy;
			dcomplex  alpha_chi1;

			bli_zcopycjs( conjx, *chi1, alpha_chi1 );
			bli_zscals( *alpha, alpha_chi1 );

			f
				(
				 conja,
				 m,
				 &alpha_chi1,
				 a1, inca,
				 y1, incy,
				 cntx
				);
		}
        
		return;
	}


	// At this point, we know that b_n is exactly equal to the fusing factor.

	a0   = a + 0*lda;
	a1   = a + 1*lda;
	a2   = a + 2*lda;
	a3   = a + 3*lda;
	a4   = a + 4*lda;
	y0   = y;

	chi0 = *( x + 0*incx );
	chi1 = *( x + 1*incx );
	chi2 = *( x + 2*incx );
	chi3 = *( x + 3*incx );
	chi4 = *( x + 4*incx );

	dcomplex *pchi0 = x + 0*incx ;
	dcomplex *pchi1 = x + 1*incx ;
	dcomplex *pchi2 = x + 2*incx ;
	dcomplex *pchi3 = x + 3*incx ;
	dcomplex *pchi4 = x + 4*incx ;

	bli_zcopycjs( conjx, *pchi0, chi0 );
	bli_zcopycjs( conjx, *pchi1, chi1 );
	bli_zcopycjs( conjx, *pchi2, chi2 );
	bli_zcopycjs( conjx, *pchi3, chi3 );
	bli_zcopycjs( conjx, *pchi4, chi4 );

	// Scale each chi scalar by alpha.
	bli_zscals( *alpha, chi0 );
	bli_zscals( *alpha, chi1 );
	bli_zscals( *alpha, chi2 );
	bli_zscals( *alpha, chi3 );
	bli_zscals( *alpha, chi4 );

	// Broadcast the (alpha*chi?) scalars to all elements of vector registers.
	chi0v.v = _mm256_broadcast_sd( &chi0.real );
	chi1v.v = _mm256_broadcast_sd( &chi1.real );
	chi2v.v = _mm256_broadcast_sd( &chi2.real );
	chi3v.v = _mm256_broadcast_sd( &chi3.real );
	chi4v.v = _mm256_broadcast_sd( &chi4.real );

	chi5v.v = _mm256_broadcast_sd( &chi0.imag );
	chi6v.v = _mm256_broadcast_sd( &chi1.imag );
	chi7v.v = _mm256_broadcast_sd( &chi2.imag );
	chi8v.v = _mm256_broadcast_sd( &chi3.imag );
	chi9v.v = _mm256_broadcast_sd( &chi4.imag );

	// If there are vectorized iterations, perform them with vector
	// instructions.
	if ( inca == 1 && incy == 1 )
	{
		// March through vectors in multiple of 4.
		for( i = 0; (i + 3) < m; i += 4 )
		{
			// Load the input values.
			r0v.v = _mm256_loadu_pd( (double*) (y0 + 0*n_elem_per_reg ));
			r1v.v = _mm256_loadu_pd( (double*) (y0 + 1*n_elem_per_reg ));

			y0v.v = _mm256_setzero_pd();
			y1v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();
			y3v.v = _mm256_setzero_pd();

			if ( bli_is_conj(conja) ){
				/**
				 * For conjugate cases imaginary part
				 * is negated.
				 */
				conjv.v = _mm256_set_pd( -1, 1, -1, 1 );
				a00v.v = _mm256_loadu_pd( (double*) (a0 + 0*n_elem_per_reg ));
				a10v.v = _mm256_loadu_pd( (double*) (a0 + 1*n_elem_per_reg ));

				a01v.v = _mm256_loadu_pd( (double*) (a1 + 0*n_elem_per_reg ));
				a11v.v = _mm256_loadu_pd( (double*) (a1 + 1*n_elem_per_reg ));

				a02v.v = _mm256_loadu_pd( (double*) (a2 + 0*n_elem_per_reg ));
				a12v.v = _mm256_loadu_pd( (double*) (a2 + 1*n_elem_per_reg ));

				a03v.v = _mm256_loadu_pd( (double*) (a3 + 0*n_elem_per_reg ));
				a13v.v = _mm256_loadu_pd( (double*) (a3 + 1*n_elem_per_reg ));

				a04v.v = _mm256_loadu_pd( (double*) (a4 + 0*n_elem_per_reg ));
				a14v.v = _mm256_loadu_pd( (double*) (a4 + 1*n_elem_per_reg ));

				a00v.v = _mm256_mul_pd(a00v.v, conjv.v);
				a10v.v = _mm256_mul_pd(a10v.v, conjv.v);
				a01v.v = _mm256_mul_pd(a01v.v, conjv.v);
				a11v.v = _mm256_mul_pd(a11v.v, conjv.v);
				a02v.v = _mm256_mul_pd(a02v.v, conjv.v);
				a12v.v = _mm256_mul_pd(a12v.v, conjv.v);
				a03v.v = _mm256_mul_pd(a03v.v, conjv.v);
				a13v.v = _mm256_mul_pd(a13v.v, conjv.v);
				a04v.v = _mm256_mul_pd(a04v.v, conjv.v);
				a14v.v = _mm256_mul_pd(a14v.v, conjv.v);
			}
			else
			{
				a00v.v = _mm256_loadu_pd( (double*) (a0 + 0*n_elem_per_reg ));
				a10v.v = _mm256_loadu_pd( (double*) (a0 + 1*n_elem_per_reg ));

				a01v.v = _mm256_loadu_pd( (double*) (a1 + 0*n_elem_per_reg ));
				a11v.v = _mm256_loadu_pd( (double*) (a1 + 1*n_elem_per_reg ));

				a02v.v = _mm256_loadu_pd( (double*) (a2 + 0*n_elem_per_reg ));
				a12v.v = _mm256_loadu_pd( (double*) (a2 + 1*n_elem_per_reg ));

				a03v.v = _mm256_loadu_pd( (double*) (a3 + 0*n_elem_per_reg ));
				a13v.v = _mm256_loadu_pd( (double*) (a3 + 1*n_elem_per_reg ));

				a04v.v = _mm256_loadu_pd( (double*) (a4 + 0*n_elem_per_reg ));
				a14v.v = _mm256_loadu_pd( (double*) (a4 + 1*n_elem_per_reg ));

			}

			// perform : y += alpha * x;
			/**
			 * chi[x]v.v holds real part of chi.
			 * chi[x]v.v holds imag part of chi.
			 * ys holds following computation:
			 *
			 *   a[xx]v.v    R1        I1       R2         I2
			 *  chi[x]v.v   chi_R     chi_R     chi_R      chi_R
			 *  chi[x]v.v   chi_I     chi_I     chi_I      chi_I
			 *    y[x]v.v   R1*chi_R  I1*chi_R  R2*chi_R  I2*chiR (compute with chi-real part)
			 *    y[x]v.v   R1*chi_I  I1*chi_I  R2*chi_I  I2*chiI (compute with chi-imag part)
			 *
			 */
			y0v.v = _mm256_mul_pd( a00v.v, chi0v.v);
			y1v.v = _mm256_mul_pd( a10v.v, chi0v.v);

			y2v.v = _mm256_mul_pd( a00v.v, chi5v.v);
			y3v.v = _mm256_mul_pd( a10v.v, chi5v.v);

			/**
			 * y0v.v & y1v.v holds computation with real part of chi.
			 * y2v.v & y3v.v holds computaion with imag part of chi.
			 * Permute will swap the positions of elements in y2v.v & y3v.v
			 * as we need to perform: [ R*R + I*I & R*I + I*R].
			 * Once dcomplex multiplication is done add the result into r0v.v
			 * r1v.v which holds axpy result of current tile which is being
			 * computed.
			 */
			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y3v.v = _mm256_permute_pd(y3v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			y1v.v = _mm256_addsub_pd(y1v.v, y3v.v);

			r0v.v = _mm256_add_pd(y0v.v, r0v.v);
			r1v.v = _mm256_add_pd(y1v.v, r1v.v);

			y0v.v = _mm256_setzero_pd();
			y1v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();
			y3v.v = _mm256_setzero_pd();

			/**
			 * Repeat the same computation as above
			 * for remaining tile.
			 */
			y0v.v = _mm256_mul_pd( a01v.v, chi1v.v );
			y1v.v = _mm256_mul_pd( a11v.v, chi1v.v );

			y2v.v = _mm256_mul_pd( a01v.v, chi6v.v );
			y3v.v = _mm256_mul_pd( a11v.v, chi6v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y3v.v = _mm256_permute_pd(y3v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			y1v.v = _mm256_addsub_pd(y1v.v, y3v.v);

			r0v.v = _mm256_add_pd(y0v.v, r0v.v);
			r1v.v = _mm256_add_pd(y1v.v, r1v.v);

			y0v.v = _mm256_setzero_pd();
			y1v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();
			y3v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a02v.v, chi2v.v);
			y1v.v = _mm256_mul_pd( a12v.v, chi2v.v);

			y2v.v = _mm256_mul_pd( a02v.v, chi7v.v );
			y3v.v = _mm256_mul_pd( a12v.v, chi7v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y3v.v = _mm256_permute_pd(y3v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			y1v.v = _mm256_addsub_pd(y1v.v, y3v.v);

			r0v.v = _mm256_add_pd(y0v.v, r0v.v);
			r1v.v = _mm256_add_pd(y1v.v, r1v.v);

			y0v.v = _mm256_setzero_pd();
			y1v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();
			y3v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a03v.v, chi3v.v );
			y1v.v = _mm256_mul_pd( a13v.v, chi3v.v );

			y2v.v = _mm256_mul_pd( a03v.v, chi8v.v );
			y3v.v = _mm256_mul_pd( a13v.v, chi8v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y3v.v = _mm256_permute_pd(y3v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			y1v.v = _mm256_addsub_pd(y1v.v, y3v.v);

			r0v.v = _mm256_add_pd(y0v.v, r0v.v);
			r1v.v = _mm256_add_pd(y1v.v, r1v.v);

			y0v.v = _mm256_setzero_pd();
			y1v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();
			y3v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a04v.v, chi4v.v );
			y1v.v = _mm256_mul_pd( a14v.v, chi4v.v );

			y2v.v = _mm256_mul_pd( a04v.v, chi9v.v );
			y3v.v = _mm256_mul_pd( a14v.v, chi9v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y3v.v = _mm256_permute_pd(y3v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			y1v.v = _mm256_addsub_pd(y1v.v, y3v.v);

			r0v.v = _mm256_add_pd(y0v.v, r0v.v);
			r1v.v = _mm256_add_pd(y1v.v, r1v.v);

			/**
			 * Final axpy compuation is available in r0v.v
			 * and r1v.v registers.
			 * Store it back into y vector.
			 */
			_mm256_storeu_pd( (double*) (y0 + 0*n_elem_per_reg), r0v.v );
			_mm256_storeu_pd( (double*) (y0 + 1*n_elem_per_reg), r1v.v );

			/**
			 * Set the pointers next vectors elements to be
			 * computed based on unroll factor.
			 */
			y0 += n_elem_per_reg * n_iter_unroll;
			a0 += n_elem_per_reg * n_iter_unroll;
			a1 += n_elem_per_reg * n_iter_unroll;
			a2 += n_elem_per_reg * n_iter_unroll;
			a3 += n_elem_per_reg * n_iter_unroll;
			a4 += n_elem_per_reg * n_iter_unroll;
		}
		// March through vectors in multiple of 2.
		for(  ; (i + 1) < m; i += 2 )
		{
			r0v.v = _mm256_loadu_pd( (double*) (y0 + 0*n_elem_per_reg ));

			y0v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();

			if ( bli_is_conj(conja) ){
				conjv.v = _mm256_set_pd( -1, 1, -1, 1 );
				a00v.v = _mm256_loadu_pd( (double*) (a0 + 0*n_elem_per_reg ));

				a01v.v = _mm256_loadu_pd( (double*) (a1 + 0*n_elem_per_reg ));

				a02v.v = _mm256_loadu_pd( (double*) (a2 + 0*n_elem_per_reg ));

				a03v.v = _mm256_loadu_pd( (double*) (a3 + 0*n_elem_per_reg ));

				a04v.v = _mm256_loadu_pd( (double*) (a4 + 0*n_elem_per_reg ));

				a00v.v = _mm256_mul_pd(a00v.v, conjv.v);
				a01v.v = _mm256_mul_pd(a01v.v, conjv.v);
				a02v.v = _mm256_mul_pd(a02v.v, conjv.v);
				a03v.v = _mm256_mul_pd(a03v.v, conjv.v);
				a04v.v = _mm256_mul_pd(a04v.v, conjv.v);
			}
			else
			{
				a00v.v = _mm256_loadu_pd( (double*) (a0 + 0*n_elem_per_reg ));

				a01v.v = _mm256_loadu_pd( (double*) (a1 + 0*n_elem_per_reg ));

				a02v.v = _mm256_loadu_pd( (double*) (a2 + 0*n_elem_per_reg ));

				a03v.v = _mm256_loadu_pd( (double*) (a3 + 0*n_elem_per_reg ));

				a04v.v = _mm256_loadu_pd( (double*) (a4 + 0*n_elem_per_reg ));

			}

			// perform : y += alpha * x;
			/**
			 * chi[x]v.v holds real part of chi.
			 * chi[x]v.v holds imag part of chi.
			 * ys holds following computation:
			 *
			 *   a[xx]v.v    R1        I1       R2         I2
			 *  chi[x]v.v   chi_R     chi_R     chi_R      chi_R
			 *  chi[x]v.v   chi_I     chi_I     chi_I      chi_I
			 *    y[x]v.v   R1*chi_R  I1*chi_R  R2*chi_R  I2*chiR (compute with chi-real part)
			 *    y[x]v.v   R1*chi_I  I1*chi_I  R2*chi_I  I2*chiI (compute with chi-imag part)
			 *
			 */
			y0v.v = _mm256_mul_pd( a00v.v, chi0v.v );
			y2v.v = _mm256_mul_pd( a00v.v, chi5v.v );

			/**
			 * y0v.v holds computation with real part of chi.
			 * y2v.v holds computaion with imag part of chi.
			 * Permute will swap the positions of elements in y2v.v.
			 * as we need to perform: [ R*R + I*I & R*I + I*R].
			 * Once dcomplex multiplication is done add the result into r0v.v
			 * which holds axpy result of current tile which is being
			 * computed.
			 */
			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			r0v.v = _mm256_add_pd(y0v.v, r0v.v);

			y0v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();

			/**
			 * Repeat the same computation as above
			 * for remaining tile.
			 */
			y0v.v = _mm256_mul_pd( a01v.v, chi1v.v );
			y2v.v = _mm256_mul_pd( a01v.v, chi6v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			r0v.v = _mm256_add_pd(y0v.v, r0v.v);

			y0v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a02v.v, chi2v.v );
			y2v.v = _mm256_mul_pd( a02v.v, chi7v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			r0v.v = _mm256_add_pd(y0v.v, r0v.v);

			y0v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a03v.v, chi3v.v );
			y2v.v = _mm256_mul_pd( a03v.v, chi8v.v );

			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			r0v.v = _mm256_add_pd(y0v.v, r0v.v);

			y0v.v = _mm256_setzero_pd();
			y2v.v = _mm256_setzero_pd();


			y0v.v = _mm256_mul_pd( a04v.v, chi4v.v );
			y2v.v = _mm256_mul_pd( a04v.v, chi9v.v );


			y2v.v = _mm256_permute_pd(y2v.v, 0x5);
			y0v.v = _mm256_addsub_pd(y0v.v, y2v.v);
			r0v.v = _mm256_add_pd(y0v.v, r0v.v);

			/**
			 * Final axpy compuation is available in r0v.v
			 * Store it back into y vector.
			 */
			_mm256_storeu_pd( (double*) (y0 + 0*n_elem_per_reg), r0v.v );

			y0 +=  n_iter_unroll;
			a0 +=  n_iter_unroll;
			a1 +=  n_iter_unroll;
			a2 +=  n_iter_unroll;
			a3 +=  n_iter_unroll;
			a4 +=  n_iter_unroll;

		}

		// If there are leftover iterations, perform them with scalar code.
		for ( ; (i + 0) < m ; ++i )
		{
			dcomplex       y0c = *y0;

			const dcomplex a0c = *a0;
			const dcomplex a1c = *a1;
			const dcomplex a2c = *a2;
			const dcomplex a3c = *a3;
			const dcomplex a4c = *a4;

			y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
			y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
			y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
			y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
			y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

			y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
			y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
			y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
			y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
			y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

			*y0 = y0c;

			a0 += 1;
			a1 += 1;
			a2 += 1;
			a3 += 1;
			a4 += 1;
			y0 += 1;
		}
	}
	else
	{
		for ( ; (i + 0) < m ; ++i )
		{
			dcomplex       y0c = *y0;

			const dcomplex a0c = *a0;
			const dcomplex a1c = *a1;
			const dcomplex a2c = *a2;
			const dcomplex a3c = *a3;
			const dcomplex a4c = *a4;

			y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
			y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
			y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
			y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
			y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

			y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
			y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
			y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
			y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
			y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

			*y0 = y0c;

			a0 += inca;
			a1 += inca;
			a2 += inca;
			a3 += inca;
			a4 += inca;
			y0 += incy;
		}

	}
}

