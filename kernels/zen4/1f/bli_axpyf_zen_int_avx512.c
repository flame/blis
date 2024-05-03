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

#include "immintrin.h"
#include "blis.h"

#if defined __clang__
    #define UNROLL_LOOP_FULL() _Pragma("clang loop unroll(full)")
#elif defined __GNUC__
    #define UNROLL_LOOP_FULL() _Pragma("GCC unroll 32")
#else
    #define UNROLL_LOOP_FULL()
#endif

#define GENTFUNC_AXPYF(FUSE_FACTOR) \
    void PASTEMAC2(daxpyf_zen_int, FUSE_FACTOR, _avx512) \
      ( \
       conj_t           conja, \
       conj_t           conjx, \
       dim_t            m, \
       dim_t            b_n, \
       double* restrict alpha, \
       double* restrict a, inc_t inca, inc_t lda, \
       double* restrict x, inc_t incx, \
       double* restrict y0, inc_t incy, \
       cntx_t* restrict cntx \
     ) \
{ \
    const dim_t         fuse_fac       = FUSE_FACTOR; \
    const dim_t         n_elem_per_reg = 8; \
    dim_t               i = 0; \
    \
    __m512d chi[fuse_fac]; \
    __m512d av[1]; \
    __m512d yv[1]; \
    double* as[fuse_fac] __attribute__((aligned(64))); \
    double* y = y0; \
    \
    /* If either dimension is zero, or if alpha is zero, return early.*/ \
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return; \
    \
    /* If b_n is not equal to the fusing factor, then perform the entire
       operation as a loop over axpyv. */ \
    if ( b_n != fuse_fac ) \
    { \
        daxpyv_ker_ft f = bli_daxpyv_zen_int_avx512; \
        \
        for ( i = 0; i < b_n; ++i ) \
        { \
            double* a1   = a + (0  )*inca + (i  )*lda; \
            double* chi1 = x + (i  )*incx; \
            double* y1   = y + (0  )*incy; \
            double  alphavchi1; \
            \
            bli_dcopycjs( conjx, *chi1, alphavchi1 ); \
            bli_dscals( *alpha, alphavchi1 ); \
            \
            f \
            ( \
              conja, \
              m, \
              &alphavchi1, \
              a1, inca, \
              y1, incy, \
              cntx \
            ); \
        } \
        return; \
    } \
    \
    /* At this point, we know that b_n is exactly equal to the fusing factor.*/ \
    UNROLL_LOOP_FULL() \
    for (dim_t ii = 0; ii < fuse_fac; ++ii) \
    { \
        as[ii] = a + (ii * lda); \
        chi[ii] = _mm512_set1_pd( (*alpha) * (*(x + ii * incx)) ); \
    } \
    /* If there are vectorized iterations, perform them with vector
     instructions.*/ \
    if ( inca == 1 && incy == 1 ) \
    { \
        __mmask8 m_mask; \
        m_mask = (1 << 8) - 1; \
        for ( ; i < m; i += 8) \
        { \
            if ( (m - i) < 8) m_mask = (1 << (m - i)) - 1; \
            yv[0] = _mm512_mask_loadu_pd( chi[0], m_mask, y ); \
             \
            UNROLL_LOOP_FULL() \
            for(int ii = 0; ii < fuse_fac; ++ii) \
            { \
                av[0] = _mm512_loadu_pd( as[ii] ); \
                as[ii] += n_elem_per_reg; \
                yv[0] = _mm512_fmadd_pd( av[0], chi[ii], yv[0]); \
            } \
            _mm512_mask_storeu_pd( (double *)(y ), m_mask, yv[0] ); \
            \
            y += n_elem_per_reg; \
        } \
    } \
    else \
    { \
        double       yc = *y; \
        double       chi_s[8]; \
         \
        UNROLL_LOOP_FULL() \
        for (dim_t ii = 0; ii < 8; ++ii) \
        { \
            chi_s[ii] = *(x + ii * incx) * *alpha; \
        } \
        for ( i = 0; (i + 0) < m ; ++i ) \
        { \
            yc = *y; \
            yc += chi_s[0] * (*as[0]); \
            yc += chi_s[1] * (*as[1]); \
            yc += chi_s[2] * (*as[2]); \
            yc += chi_s[3] * (*as[3]); \
            yc += chi_s[4] * (*as[4]); \
            yc += chi_s[5] * (*as[5]); \
            yc += chi_s[6] * (*as[6]); \
            yc += chi_s[7] * (*as[7]); \
            \
            *y = yc; \
            \
            as[0] += inca; \
            as[1] += inca; \
            as[2] += inca; \
            as[3] += inca; \
            as[4] += inca; \
            as[5] += inca; \
            as[6] += inca; \
            as[7] += inca; \
            \
            y += incy;  \
        } \
    } \
} \

// Generate two axpyf kernels with fuse_factor = 8 and 32
GENTFUNC_AXPYF(8)
GENTFUNC_AXPYF(32)

#ifdef BLIS_ENABLE_OPENMP
/*
* Multihreaded AVX512 DAXPYF kernel with fuse factor 32
*/
void bli_daxpyf_zen_int32_avx512_mt
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
    /*
      Initializing the number of thread to one
      to avoid compiler warnings
    */
    dim_t nt = 1;
    /*
      For the given problem size and architecture, the function
      returns the optimum number of threads with AOCL dynamic enabled
      else it returns the number of threads requested by the user.
    */
    bli_nthreads_l1
    (
        BLIS_AXPYF_KER,
        BLIS_DOUBLE,
        BLIS_DOUBLE,
        bli_arch_query_id(),
        m,
        &nt
    );

    _Pragma("omp parallel num_threads(nt)")
    {
        const dim_t tid = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        // if num threads requested and num thread available
        // is not same then use single thread
        if( nt_real != nt )
        {
            if( tid == 0 )
            {
                bli_daxpyf_zen_int32_avx512
                (
                    conja,
                    conjx,
                    m,
                    b_n,
                    alpha,
                    a,
                    inca,
                    lda,
                    x,
                    incx,
                    y,
                    incy,
                    cntx
                );
            }
        }
        else
        {
            dim_t job_per_thread, offset;

            // Obtain the job-size and region for compute
            bli_normfv_thread_partition( m, nt_real, &offset, &job_per_thread, 32, incy, tid );

            // Calculate y_start and a_start for current thread
            double* restrict y_start = y + offset;
            double* restrict a_start = a + offset;

            // call axpyf kernel
            bli_daxpyf_zen_int32_avx512
            (
                conja,
                conjx,
                job_per_thread,
                b_n,
                alpha,
                a_start,
                inca,
                lda,
                x,
                incx,
                y_start,
                incy,
                cntx
            );
        }
    }
}
#endif
