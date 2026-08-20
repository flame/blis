/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#define ARCH_SIMD_BITS  256
#define GEMV_ARCH_SUFFIX zen_int
#define GEMV_BLK_SUFFIX_N(ch, MR, NR)  PASTEMAC5(ch, gemv_n_block_, MR, _, NR, _avx2)
#include "bli_pp_common.h"

#ifdef BLIS_ENABLE_OPENMP
#include <omp.h>
#endif

#include "bli_gemv_n_impl.h"

/*
 * ─── Complex types (c/z): macro-generated kernels + inlined interface ────────
 *
 * The compute layer stays macro-generated: the micro-kernels + dispatch table
 * (GENERATE_<ch>_KERNELS_<MR>_N) and the single-thread tiled caller
 * (GENT_GEMV_CALLER). The threading wrapper (_mt) and the public entry-point are
 * written out as explicit type-specific C — mirroring GENT_N_GEMV_M_DIM and
 * GENERATE_ROOT_KERNEL respectively — so the interface is readable/debuggable
 * while the kernels remain generated.
 *
 * Complex uses the N-kernel only: MT is M-split (no N-split/reduction), and the
 * entry buffers a conjugated x (conjx) and/or a contiguous y (incy != 1) so the
 * kernel always sees unit-stride, non-conjugated inputs.
 */

// ═══ scomplex (c): MR_N=20, NR_N=5 ═══════════════════════════════════════════
// expands to: the N-direction micro-kernel family bli_cgemv_n_block_*_avx2 (one
//             per row sub-tile) + the static dispatch table bli_cgemv_n_ker_fp_20_5
GENERATE_c_KERNELS_20_N(scomplex, c, 20, 5);
// expands to: bli_cgemv_n_zen_int_20x5 (single-thread N-direction tiled caller)
GENT_GEMV_CALLER(scomplex, c, 20, 5, n)

#ifdef BLIS_ENABLE_OPENMP
// MT wrapper: split M (output rows) across threads; disjoint y rows, no reduction.
void bli_cgemv_n_zen_int_20x5_mt
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       scomplex* alpha, scomplex* a, inc_t rs_a, inc_t cs_a,
       scomplex* x, inc_t incx, scomplex* beta,
       scomplex* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_cgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_SCOMPLEX, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_cgemv_n_zen_int_20x5( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
        return;
    }

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );
        bli_cgemv_n_zen_int_20x5( transa, conjx, job_per_thread, n, alpha,
                                  a + thread_start * rs_a, rs_a, cs_a,
                                  x, incx, beta,
                                  y + thread_start * incy, incy, cntx );
    }
}
#endif

// Public entry-point: alpha==0 scale-only, conj-x buffering, incy!=1 y buffering,
// then ST vs MT dispatch by problem size.
void bli_cgemv_n_zen_int
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       scomplex* alpha, scomplex* a, inc_t rs_a, inc_t cs_a,
       scomplex* x, inc_t incx, scomplex* beta,
       scomplex* y, inc_t incy, cntx_t* cntx
     )
{
    void (*ker_ft)( trans_t, conj_t, dim_t, dim_t, scomplex*, scomplex*, inc_t, inc_t,
                    scomplex*, inc_t, scomplex*, scomplex*, inc_t, cntx_t* ) = NULL;
    rntm_t    rntm;
    mem_t     mem_bufX, mem_bufY;
    inc_t     temp_incx = incx, temp_incy = incy;
    scomplex* x_temp = x;
    scomplex* y_temp = y;
    bool      is_y_temp_buf_created = FALSE;
    ccopyv_ker_ft copyv_kr_ptr = NULL;

    // alpha == 0: y = beta * y (scale only), skip the multiply entirely.
    if ( bli_ceq0( *alpha ) )
    {
        cscalv_ker_ft scalv_kr_ptr =
            bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_SCALV_KER, cntx );
        scalv_kr_ptr( BLIS_NO_CONJUGATE, m, beta, y, incy, cntx );
        return;
    }

    // conjx: copy-and-conjugate x into a contiguous temp so the kernel always
    // sees a plain, unit-stride x.
    const bool need_conj = bli_is_conj( conjx );
    if ( need_conj )
    {
        mem_bufX.pblk.buf = NULL;   mem_bufX.pblk.block_size = 0;
        mem_bufX.buf_type = 0;      mem_bufX.size = 0;
        mem_bufX.pool = NULL;

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        bli_pba_acquire_m( &rntm, n * sizeof( scomplex ),
                           BLIS_BUFFER_FOR_B_PANEL, &mem_bufX );
        if ( bli_mem_is_alloc( &mem_bufX ) )
        {
            x_temp    = bli_mem_buffer( &mem_bufX );
            temp_incx = 1;
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            copyv_kr_ptr = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_COPYV_KER, cntx );
            copyv_kr_ptr( BLIS_CONJUGATE, n, x, incx, x_temp, temp_incx, cntx );
        }
        else
        {
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            bli_cgemv_unb_var2( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
            return;
        }
    }

    // incy != 1: buffer y into a contiguous temp (packed by beta) so the kernel
    // always sees incy == 1; unpack after the multiply.
    if ( incy != 1 )
    {
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;
        if ( !need_conj )
        {
            bli_rntm_init_from_global( &rntm );
            bli_rntm_set_num_threads_only( 1, &rntm );
            bli_pba_rntm_set_pba( &rntm );
        }
        bli_pba_acquire_m( &rntm, m * sizeof( scomplex ),
                           BLIS_BUFFER_FOR_B_PANEL, &mem_bufY );
        if ( bli_mem_is_alloc( &mem_bufY ) )
        {
            y_temp    = bli_mem_buffer( &mem_bufY );
            temp_incy = 1;
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            if ( copyv_kr_ptr == NULL )
                copyv_kr_ptr = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_COPYV_KER, cntx );
            if ( !bli_ceq0( *beta ) )
                copyv_kr_ptr( BLIS_NO_CONJUGATE, m, y, incy, y_temp, temp_incy, cntx );
            is_y_temp_buf_created = TRUE;
        }
        else
        {
            bli_cgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a,
                               x_temp, temp_incx, beta, y, incy, NULL );
            if ( x_temp != x )
                bli_pba_release( &rntm, &mem_bufX );
            return;
        }
    }

#if defined(BLIS_ENABLE_OPENMP)
    ker_ft = ( m * n < 1800 ) ? bli_cgemv_n_zen_int_20x5
                              : bli_cgemv_n_zen_int_20x5_mt;
#else
    ker_ft = bli_cgemv_n_zen_int_20x5;
#endif

    ker_ft( transa, BLIS_NO_CONJUGATE, m, n, alpha, a, rs_a, cs_a,
            x_temp, temp_incx, beta, y_temp, temp_incy, cntx );

    if ( is_y_temp_buf_created )
    {
        copyv_kr_ptr( BLIS_NO_CONJUGATE, m, y_temp, temp_incy, y, incy, cntx );
        bli_pba_release( &rntm, &mem_bufY );
    }
    if ( x_temp != x )
        bli_pba_release( &rntm, &mem_bufX );
}

// ═══ dcomplex (z): MR_N=10, NR_N=5 ═══════════════════════════════════════════
// expands to: the N-direction micro-kernel family bli_zgemv_n_block_*_avx2 (one
//             per row sub-tile) + the static dispatch table bli_zgemv_n_ker_fp_10_5
GENERATE_z_KERNELS_10_N(dcomplex, z, 10, 5);
// expands to: bli_zgemv_n_zen_int_10x5 (single-thread N-direction tiled caller)
GENT_GEMV_CALLER(dcomplex, z, 10, 5, n)

#ifdef BLIS_ENABLE_OPENMP
// MT wrapper: split M (output rows) across threads; disjoint y rows, no reduction.
void bli_zgemv_n_zen_int_10x5_mt
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       dcomplex* alpha, dcomplex* a, inc_t rs_a, inc_t cs_a,
       dcomplex* x, inc_t incx, dcomplex* beta,
       dcomplex* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_zgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_DCOMPLEX, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_zgemv_n_zen_int_10x5( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
        return;
    }

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );
        bli_zgemv_n_zen_int_10x5( transa, conjx, job_per_thread, n, alpha,
                                  a + thread_start * rs_a, rs_a, cs_a,
                                  x, incx, beta,
                                  y + thread_start * incy, incy, cntx );
    }
}
#endif

// Public entry-point: alpha==0 scale-only, conj-x buffering, incy!=1 y buffering,
// then ST vs MT dispatch by problem size.
void bli_zgemv_n_zen_int
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       dcomplex* alpha, dcomplex* a, inc_t rs_a, inc_t cs_a,
       dcomplex* x, inc_t incx, dcomplex* beta,
       dcomplex* y, inc_t incy, cntx_t* cntx
     )
{
    void (*ker_ft)( trans_t, conj_t, dim_t, dim_t, dcomplex*, dcomplex*, inc_t, inc_t,
                    dcomplex*, inc_t, dcomplex*, dcomplex*, inc_t, cntx_t* ) = NULL;
    rntm_t    rntm;
    mem_t     mem_bufX, mem_bufY;
    inc_t     temp_incx = incx, temp_incy = incy;
    dcomplex* x_temp = x;
    dcomplex* y_temp = y;
    bool      is_y_temp_buf_created = FALSE;
    zcopyv_ker_ft copyv_kr_ptr = NULL;

    // alpha == 0: y = beta * y (scale only), skip the multiply entirely.
    if ( bli_zeq0( *alpha ) )
    {
        zscalv_ker_ft scalv_kr_ptr =
            bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx );
        scalv_kr_ptr( BLIS_NO_CONJUGATE, m, beta, y, incy, cntx );
        return;
    }

    // conjx: copy-and-conjugate x into a contiguous temp so the kernel always
    // sees a plain, unit-stride x.
    const bool need_conj = bli_is_conj( conjx );
    if ( need_conj )
    {
        mem_bufX.pblk.buf = NULL;   mem_bufX.pblk.block_size = 0;
        mem_bufX.buf_type = 0;      mem_bufX.size = 0;
        mem_bufX.pool = NULL;

        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_pba_rntm_set_pba( &rntm );

        bli_pba_acquire_m( &rntm, n * sizeof( dcomplex ),
                           BLIS_BUFFER_FOR_B_PANEL, &mem_bufX );
        if ( bli_mem_is_alloc( &mem_bufX ) )
        {
            x_temp    = bli_mem_buffer( &mem_bufX );
            temp_incx = 1;
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            copyv_kr_ptr = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_COPYV_KER, cntx );
            copyv_kr_ptr( BLIS_CONJUGATE, n, x, incx, x_temp, temp_incx, cntx );
        }
        else
        {
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            bli_zgemv_unb_var2( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
            return;
        }
    }

    // incy != 1: buffer y into a contiguous temp (packed by beta) so the kernel
    // always sees incy == 1; unpack after the multiply.
    if ( incy != 1 )
    {
        mem_bufY.pblk.buf = NULL;   mem_bufY.pblk.block_size = 0;
        mem_bufY.buf_type = 0;      mem_bufY.size = 0;
        mem_bufY.pool = NULL;
        if ( !need_conj )
        {
            bli_rntm_init_from_global( &rntm );
            bli_rntm_set_num_threads_only( 1, &rntm );
            bli_pba_rntm_set_pba( &rntm );
        }
        bli_pba_acquire_m( &rntm, m * sizeof( dcomplex ),
                           BLIS_BUFFER_FOR_B_PANEL, &mem_bufY );
        if ( bli_mem_is_alloc( &mem_bufY ) )
        {
            y_temp    = bli_mem_buffer( &mem_bufY );
            temp_incy = 1;
            if ( cntx == NULL ) cntx = bli_gks_query_cntx();
            if ( copyv_kr_ptr == NULL )
                copyv_kr_ptr = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_COPYV_KER, cntx );
            if ( !bli_zeq0( *beta ) )
                copyv_kr_ptr( BLIS_NO_CONJUGATE, m, y, incy, y_temp, temp_incy, cntx );
            is_y_temp_buf_created = TRUE;
        }
        else
        {
            bli_zgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a,
                               x_temp, temp_incx, beta, y, incy, NULL );
            if ( x_temp != x )
                bli_pba_release( &rntm, &mem_bufX );
            return;
        }
    }

#if defined(BLIS_ENABLE_OPENMP)
    ker_ft = ( m * n < 1800 ) ? bli_zgemv_n_zen_int_10x5
                              : bli_zgemv_n_zen_int_10x5_mt;
#else
    ker_ft = bli_zgemv_n_zen_int_10x5;
#endif

    ker_ft( transa, BLIS_NO_CONJUGATE, m, n, alpha, a, rs_a, cs_a,
            x_temp, temp_incx, beta, y_temp, temp_incy, cntx );

    if ( is_y_temp_buf_created )
    {
        copyv_kr_ptr( BLIS_NO_CONJUGATE, m, y_temp, temp_incy, y, incy, cntx );
        bli_pba_release( &rntm, &mem_bufY );
    }
    if ( x_temp != x )
        bli_pba_release( &rntm, &mem_bufX );
}

// expands to: N micro-kernel family bli_dgemv_n_block_*_avx2 + table bli_dgemv_n_ker_fp_20_4
GENERATE_d_KERNELS_20_N(double, d, 20, 4);
// expands to: bli_dgemv_n_zen_int_20x4 (single-thread N-direction tiled caller)
GENT_GEMV_CALLER(double, d, 20, 4, n)
// expands to: bli_dgemv_n_zen_int_20x4_mt (M-split OpenMP wrapper; empty if !OpenMP)
MT_KERNEL_SIGNATURE(double, d, 20, 4)

// expands to: M micro-kernel family bli_dgemv_m_block_*_avx2 + table bli_dgemv_m_ker_fp_20_4
GENERATE_d_KERNELS_20_NM(double, d, 20, 4)
// expands to: bli_dgemv_m_zen_int_20x4 (single-thread M-direction tiled caller)
GENT_GEMV_CALLER(double, d, 20, 4, m)

// expands to: N micro-kernel family bli_sgemv_n_block_*_avx2 + table bli_sgemv_n_ker_fp_40_4
GENERATE_s_KERNELS_40_N(float, s, 40, 4);
// expands to: bli_sgemv_n_zen_int_40x4 (single-thread N-direction tiled caller)
GENT_GEMV_CALLER(float, s, 40, 4, n)
// expands to: bli_sgemv_n_zen_int_40x4_mt (M-split OpenMP wrapper; empty if !OpenMP)
MT_KERNEL_SIGNATURE(float, s, 40, 4)

// expands to: M micro-kernel family bli_sgemv_m_block_*_avx2 + table bli_sgemv_m_ker_fp_40_4
GENERATE_s_KERNELS_40_NM(float, s, 40, 4)
// expands to: bli_sgemv_m_zen_int_40x4 (single-thread M-direction tiled caller)
GENT_GEMV_CALLER(float, s, 40, 4, m)


/*
 * ─── Real-type (s/d) control layer: fully inlined, self-contained per type ────
 *
 * The single-thread size dispatch (_st), the multi-threaded row/column split
 * wrappers (_mt_Mdiv / _mt_Ndiv), and the public entry-point are written out
 * directly here as type-specific C — no vtable, no shared void* core. Each calls
 * the concrete AVX2 tiled callers / addv / ref generated above.
 *
 * Definition order (_st → _mt_Mdiv → _mt_Ndiv → entry) is chosen so every
 * intra-file callee is defined above its caller; no forward declarations needed.
 * (Complex types c/z stay on the GENERATE_KERNEL macro path above.)
 */

// ── double ───────────────────────────────────────────────────────────────────

// Single-thread size dispatch: M-direction caller for small problems (byte
// threshold auto-scales with the datatype), N-direction caller otherwise.
void bli_dgemv_n_zen_int_st
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       double* alpha, double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx, double* beta,
       double* y, inc_t incy, cntx_t* cntx
     )
{
    if ( (dim_t)( m * n ) * (dim_t)sizeof( double ) < GEMV_N_CTRL_THRESH_BYTES )
        bli_dgemv_m_zen_int_20x4( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
    else
        bli_dgemv_n_zen_int_20x4( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
}

// Multi-thread, row (M) split: each thread owns a disjoint block of y rows and
// runs the ST dispatcher on it (disjoint writes, no reduction).
void bli_dgemv_m_zen_int_20x4_mt_Mdiv
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       double* alpha, double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx, double* beta,
       double* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_dgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_DOUBLE, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }

#ifdef BLIS_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );
        bli_dgemv_n_zen_int_st( transa, conjx, job_per_thread, n, alpha,
                                a + thread_start * rs_a, rs_a, cs_a,
                                x, incx, beta,
                                y + thread_start * incy, incy, cntx );
    }
#else
    bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
#endif
}

// Multi-thread, column (N) split + reduction: each thread computes a partial y
// over a column slice; thread 0 writes into y, threads 1..nt-1 into per-thread
// scratch (beta=0). A serial addv loop then sums the partials into y. Scratch is
// one allocation: [ dim_t jobs[nt] ][ (nt-1) partial-y buffers of m*incy ].
void bli_dgemv_m_zen_int_20x4_mt_Ndiv
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       double* alpha, double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx, double* beta,
       double* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_dgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_DOUBLE, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }

#ifdef BLIS_ENABLE_OPENMP
    rntm_t rntm;
    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_pba_rntm_set_pba( &rntm );

    const size_t jobs_bytes = (size_t)nt * sizeof( dim_t );
    const size_t part_bytes = (size_t)m * (size_t)incy * (size_t)( nt - 1 ) * sizeof( double );
    mem_t local_mem_buf = { 0 };
    bli_pba_acquire_m( &rntm, jobs_bytes + part_bytes,
                       BLIS_BITVAL_BUFFER_FOR_GEN_USE, &local_mem_buf );

    // Total allocation failure: fall back to the reference kernel.
    if ( !bli_mem_is_alloc( &local_mem_buf ) )
    {
        bli_dgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    // Buffer allocated but unusable: release and fall back to the row-split path.
    void* temp_mem = bli_mem_buffer( &local_mem_buf );
    if ( local_mem_buf.size < jobs_bytes + part_bytes || !temp_mem )
    {
        if ( bli_mem_is_alloc( &local_mem_buf ) ) bli_pba_release( &rntm, &local_mem_buf );
        bli_dgemv_m_zen_int_20x4_mt_Mdiv( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                          x, incx, beta, y, incy, cntx );
        return;
    }

    dim_t*  jobs     = (dim_t*)temp_mem;
    double* partials = (double*)( (char*)temp_mem + jobs_bytes );
    memset( temp_mem, 0, jobs_bytes + part_bytes );

    double zero = 0.0;  // beta=0 for the partial-y threads

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( n, nt_real, &thread_start, &job_per_thread, tid );

        double* mem   = y;
        double* beta_ = beta;
        if ( tid != 0 )
        {
            mem   = partials + (size_t)( tid - 1 ) * (size_t)m * (size_t)incy;
            beta_ = &zero;
        }
        jobs[ tid ] = job_per_thread;

        bli_dgemv_n_zen_int_st( transa, conjx, m, job_per_thread, alpha,
                                a + thread_start * cs_a, rs_a, cs_a,
                                x + thread_start * incx, incx,
                                beta_, mem, incy, cntx );
    }

    for ( dim_t i = 1; i < nt; ++i )
    {
        if ( jobs[ i ] == 0 ) continue;
        double* partial = partials + (size_t)( i - 1 ) * (size_t)m * (size_t)incy;
        bli_daddv_zen_int( BLIS_NO_CONJUGATE, m, partial, incy, y, incy, cntx );
    }

    if ( bli_mem_is_alloc( &local_mem_buf ) ) bli_pba_release( &rntm, &local_mem_buf );
#else
    bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
#endif
}

void bli_dgemv_n_zen_int
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       double* alpha, double* a, inc_t rs_a, inc_t cs_a,
       double* x, inc_t incx, double* beta,
       double* y, inc_t incy, cntx_t* cntx
     )
{
    void (*ker_ft)( trans_t, conj_t, dim_t, dim_t, double*, double*, inc_t, inc_t,
                    double*, inc_t, double*, double*, inc_t, cntx_t* ) = NULL;

    // alpha == 0: y = beta * y (scale only), skip the multiply entirely.
    if ( bli_deq0( *alpha ) )
    {
        dscalv_ker_ft scalv_kr_ptr =
            bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_SCALV_KER, cntx );
        scalv_kr_ptr( BLIS_NO_CONJUGATE, m, beta, y, incy, cntx );
        return;
    }

#if defined(AOCL_DYNAMIC)
    // Small problem: skip the MT decision and dispatch straight to ST.
    if ( (dim_t)( m * n ) * (dim_t)sizeof( double ) < GEMV_N_CTRL_THRESH_BYTES )
    {
        bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }
#endif

#if defined(BLIS_ENABLE_OPENMP)
    if ( ( m < GEMV_N_CTRL_MT_THRESH_M ) ||
         ( (dim_t)( m * n ) >= GEMV_N_CTRL_MT_THRESH_SIZE && ( m / n ) < 10000 ) )
        ker_ft = bli_dgemv_m_zen_int_20x4_mt_Ndiv;
    else
        ker_ft = bli_dgemv_m_zen_int_20x4_mt_Mdiv;
#else
    bli_dgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
    return;
#endif

    // Strided y or transpose: hand off to the M-direction tiled caller.
    if ( incy != 1 || transa != BLIS_NO_TRANSPOSE )
        ker_ft = bli_dgemv_m_zen_int_20x4;

    ker_ft( transa, conjx, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx );
}

// ── float ────────────────────────────────────────────────────────────────────
void bli_sgemv_n_zen_int_st
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       float* alpha, float* a, inc_t rs_a, inc_t cs_a,
       float* x, inc_t incx, float* beta,
       float* y, inc_t incy, cntx_t* cntx
     )
{
    if ( (dim_t)( m * n ) * (dim_t)sizeof( float ) < GEMV_N_CTRL_THRESH_BYTES )
        bli_sgemv_m_zen_int_40x4( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
    else
        bli_sgemv_n_zen_int_40x4( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                  x, incx, beta, y, incy, cntx );
}

void bli_sgemv_m_zen_int_40x4_mt_Mdiv
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       float* alpha, float* a, inc_t rs_a, inc_t cs_a,
       float* x, inc_t incx, float* beta,
       float* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_sgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_FLOAT, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }

#ifdef BLIS_ENABLE_OPENMP
    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( m, nt_real, &thread_start, &job_per_thread, tid );
        bli_sgemv_n_zen_int_st( transa, conjx, job_per_thread, n, alpha,
                                a + thread_start * rs_a, rs_a, cs_a,
                                x, incx, beta,
                                y + thread_start * incy, incy, cntx );
    }
#else
    bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
#endif
}

void bli_sgemv_m_zen_int_40x4_mt_Ndiv
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       float* alpha, float* a, inc_t rs_a, inc_t cs_a,
       float* x, inc_t incx, float* beta,
       float* y, inc_t incy, cntx_t* cntx
     )
{
    if ( ( ( rs_a != 1 ) && ( cs_a != 1 ) ) || transa != BLIS_NO_TRANSPOSE || incy != 1 )
    {
        bli_sgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    dim_t nt = 1;
    bli_nthreads_l2( BLIS_GEMV_KER, BLIS_FLOAT, BLIS_NO_TRANSPOSE,
                     bli_arch_query_id_internal(), m, n, &nt );

    if ( nt == 1 )
    {
        bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }

#ifdef BLIS_ENABLE_OPENMP
    rntm_t rntm;
    bli_rntm_init_from_global( &rntm );
    bli_rntm_set_num_threads_only( 1, &rntm );
    bli_pba_rntm_set_pba( &rntm );

    const size_t jobs_bytes = (size_t)nt * sizeof( dim_t );
    const size_t part_bytes = (size_t)m * (size_t)incy * (size_t)( nt - 1 ) * sizeof( float );
    mem_t local_mem_buf = { 0 };
    bli_pba_acquire_m( &rntm, jobs_bytes + part_bytes,
                       BLIS_BITVAL_BUFFER_FOR_GEN_USE, &local_mem_buf );

    if ( !bli_mem_is_alloc( &local_mem_buf ) )
    {
        bli_sgemv_zen_ref( transa, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, NULL );
        return;
    }

    void* temp_mem = bli_mem_buffer( &local_mem_buf );
    if ( local_mem_buf.size < jobs_bytes + part_bytes || !temp_mem )
    {
        if ( bli_mem_is_alloc( &local_mem_buf ) ) bli_pba_release( &rntm, &local_mem_buf );
        bli_sgemv_m_zen_int_40x4_mt_Mdiv( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                          x, incx, beta, y, incy, cntx );
        return;
    }

    dim_t* jobs     = (dim_t*)temp_mem;
    float* partials = (float*)( (char*)temp_mem + jobs_bytes );
    memset( temp_mem, 0, jobs_bytes + part_bytes );

    float zero = 0.0f;  // beta=0 for the partial-y threads

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t job_per_thread = m, thread_start = 0;
        const dim_t tid     = omp_get_thread_num();
        const dim_t nt_real = omp_get_num_threads();
        bli_thread_vector_partition( n, nt_real, &thread_start, &job_per_thread, tid );

        float* mem   = y;
        float* beta_ = beta;
        if ( tid != 0 )
        {
            mem   = partials + (size_t)( tid - 1 ) * (size_t)m * (size_t)incy;
            beta_ = &zero;
        }
        jobs[ tid ] = job_per_thread;

        bli_sgemv_n_zen_int_st( transa, conjx, m, job_per_thread, alpha,
                                a + thread_start * cs_a, rs_a, cs_a,
                                x + thread_start * incx, incx,
                                beta_, mem, incy, cntx );
    }

    for ( dim_t i = 1; i < nt; ++i )
    {
        if ( jobs[ i ] == 0 ) continue;
        float* partial = partials + (size_t)( i - 1 ) * (size_t)m * (size_t)incy;
        bli_saddv_zen_int( BLIS_NO_CONJUGATE, m, partial, incy, y, incy, cntx );
    }

    if ( bli_mem_is_alloc( &local_mem_buf ) ) bli_pba_release( &rntm, &local_mem_buf );
#else
    bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
#endif
}

void bli_sgemv_n_zen_int
     (
       trans_t transa, conj_t conjx, dim_t m, dim_t n,
       float* alpha, float* a, inc_t rs_a, inc_t cs_a,
       float* x, inc_t incx, float* beta,
       float* y, inc_t incy, cntx_t* cntx
     )
{
    void (*ker_ft)( trans_t, conj_t, dim_t, dim_t, float*, float*, inc_t, inc_t,
                    float*, inc_t, float*, float*, inc_t, cntx_t* ) = NULL;

    // alpha == 0: y = beta * y (scale only), skip the multiply entirely.
    if ( bli_seq0( *alpha ) )
    {
        sscalv_ker_ft scalv_kr_ptr =
            bli_cntx_get_l1v_ker_dt( BLIS_FLOAT, BLIS_SCALV_KER, cntx );
        scalv_kr_ptr( BLIS_NO_CONJUGATE, m, beta, y, incy, cntx );
        return;
    }

#if defined(AOCL_DYNAMIC)
    // Small problem: skip the MT decision and dispatch straight to ST.
    if ( (dim_t)( m * n ) * (dim_t)sizeof( float ) < GEMV_N_CTRL_THRESH_BYTES )
    {
        bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                                x, incx, beta, y, incy, cntx );
        return;
    }
#endif

#if defined(BLIS_ENABLE_OPENMP)
    if ( ( m < GEMV_N_CTRL_MT_THRESH_M ) ||
         ( (dim_t)( m * n ) >= GEMV_N_CTRL_MT_THRESH_SIZE && ( m / n ) < 10000 ) )
        ker_ft = bli_sgemv_m_zen_int_40x4_mt_Ndiv;
    else
        ker_ft = bli_sgemv_m_zen_int_40x4_mt_Mdiv;
#else
    bli_sgemv_n_zen_int_st( transa, conjx, m, n, alpha, a, rs_a, cs_a,
                            x, incx, beta, y, incy, cntx );
    return;
#endif

    // Strided y or transpose: hand off to the M-direction tiled caller.
    if ( incy != 1 || transa != BLIS_NO_TRANSPOSE )
        ker_ft = bli_sgemv_m_zen_int_40x4;

    ker_ft( transa, conjx, m, n, alpha, a, rs_a, cs_a, x, incx, beta, y, incy, cntx );
}
