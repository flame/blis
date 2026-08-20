/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

// -----------------------------------------------------------------------------
// Per (arch, datatype) dispatch macros.
//
// AVX512 and AVX2 paths call the t-kernel directly and return -- the
// kernel handles its own ST/MT dispatch and non-unit incx internally, so the
// post-switch wrapper (packing + OpenMP) is bypassed.
//
// Guard on bli_does_trans(transa): var1 can be entered for conjA+no-transpose
// (e.g. BLIS_CONJ_NO_TRANSPOSE), which must NOT be handled by the T-kernel.
// Conjugate-transpose is a true transpose case and is handled by the T-kernel.
// Fall through to the dotxf default for non-transpose transa values.
// -----------------------------------------------------------------------------

#if defined(BLIS_KERNELS_ZEN4)

#define ZEN4_s \
    if ( bli_does_trans( transa ) ) { \
    bli_sgemv_t_zen4_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN4_d \
    if ( bli_does_trans( transa ) ) { \
    bli_dgemv_t_zen4_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN4_c \
    if ( bli_does_trans( transa ) ) { \
    bli_cgemv_t_zen4_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN4_z \
    if ( bli_does_trans( transa ) ) { \
    bli_zgemv_t_zen4_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#else
#define ZEN4_s
#define ZEN4_d
#define ZEN4_c
#define ZEN4_z
#endif

#define ZEN_s \
    if ( bli_does_trans( transa ) ) { \
    bli_sgemv_t_zen_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN_d \
    if ( bli_does_trans( transa ) ) { \
    bli_dgemv_t_zen_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN_c \
    if ( bli_does_trans( transa ) ) { \
    bli_cgemv_t_zen_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

#define ZEN_z \
    if ( bli_does_trans( transa ) ) { \
    bli_zgemv_t_zen_int \
    ( \
        transa, conjx, m0, n0, alpha, \
        a, inca, lda, \
        x, incx, \
        beta, y, incy, cntx \
    ); \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
    }

// ZEN6/ZEN5/ZEN4 map to the AVX-512 (ZEN4) kernels and ZEN3/ZEN2/ZEN map to
// the AVX2 (ZEN) kernels; this grouping is now expressed via
// bli_arch_isa_tier() and the two tier cases in the switch below, so the
// per-arch alias macros (ZEN6_*, ZEN5_*, ZEN3_*, ZEN2_*) are no longer needed.


// -----------------------------------------------------------------------------

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       trans_t transa, \
       conj_t  conjx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3) \
\
    dim_t  i; \
    dim_t  f  = 0; \
    dim_t  m0 = m, n0 = n; \
    inc_t  lda = cs_a, inca = rs_a; \
    conj_t conja; \
\
    ctype *a_buf = a; \
    ctype *x_buf = x; \
    ctype *y_buf = y; \
\
    /* Invoking the reference kernel to handle general stride. The reference \
       kernel has no conjx parameter and never conjugates x, so when \
       conjugation of x is requested we first copy-conjugate x into a \
       unit-stride temporary buffer and pass that to the reference kernel. \
       (transa still carries any conjugation of A, handled inside the ref.) */ \
    if ( ( rs_a != 1 ) && ( cs_a != 1 ) ) \
    { \
        ctype*  x_gs          = x; \
        inc_t   incx_gs       = incx; \
        bool    x_gs_buffered = FALSE; \
        mem_t   mem_bufX_gs; \
        rntm_t  rntm_gs; \
\
        if ( bli_is_conj( conjx ) ) \
        { \
            const num_t dt_gs    = PASTEMAC(ch,type); \
            const dim_t x_len_gs = bli_does_notrans( transa ) ? n : m; \
            cntx_t*     cntx_gs  = cntx; \
\
            mem_bufX_gs.pblk.buf = NULL; mem_bufX_gs.pblk.block_size = 0; \
            mem_bufX_gs.buf_type = 0;    mem_bufX_gs.size = 0; \
            mem_bufX_gs.pool     = NULL; \
\
            bli_rntm_init_from_global( &rntm_gs ); \
            bli_rntm_set_num_threads_only( 1, &rntm_gs ); \
            bli_pba_rntm_set_pba( &rntm_gs ); \
            bli_pba_acquire_m( &rntm_gs, x_len_gs * sizeof( ctype ), \
                               BLIS_BUFFER_FOR_B_PANEL, &mem_bufX_gs ); \
\
            if ( bli_mem_is_alloc( &mem_bufX_gs ) ) \
            { \
                x_gs    = bli_mem_buffer( &mem_bufX_gs ); \
                incx_gs = 1; \
                if ( cntx_gs == NULL ) cntx_gs = bli_gks_query_cntx(); \
                PASTECH(ch,copyv_ker_ft) copyv_kr_ptr_gs = \
                    bli_cntx_get_l1v_ker_dt( dt_gs, BLIS_COPYV_KER, cntx_gs ); \
                copyv_kr_ptr_gs( BLIS_CONJUGATE, x_len_gs, x, incx, \
                                 x_gs, incx_gs, cntx_gs ); \
                x_gs_buffered = TRUE; \
            } \
        } \
\
        PASTEMAC(ch,gemv_zen_ref) \
        ( \
            transa, m, n, alpha, \
            a, rs_a, cs_a, \
            x_gs, incx_gs, \
            beta, \
            y, incy, NULL \
        ); \
\
        if ( x_gs_buffered ) \
            bli_pba_release( &rntm_gs, &mem_bufX_gs ); \
\
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
        return; \
    } \
\
    /* This kernel is dot-based. When op(A) = n with row-storage we compute
       y[i] = <A(i,:), x>; when op(A) = t with col-storage we compute
       y[i] = <A(:,i), x>. The dotxf kernel always walks columns of A, so we
       interchange the leading dim / inc / m / n via bli_set_dims_incs_with_trans. */ \
    bli_set_dims_incs_with_trans( transa, \
                                  m, n, rs_a, cs_a, \
                                  &n0, &m0, &lda, &inca ); \
\
    conja = bli_extract_conj( transa ); \
\
    /* Fatbinary config amdzen when run on non-AMD x86 will query for
       AVX512/AVX2 support and report zen4/zen5 or zen3 accordingly. */ \
    arch_t      arch_id = bli_arch_query_id_internal(); \
    const num_t dt      = PASTEMAC(ch,type); \
\
    /* Dispatch on ISA capability tier rather than on individual arch_id \
       values. bli_arch_isa_tier() is the single place that maps a Zen \
       arch to its tier, so new Zen parts only need to be added there. */ \
    switch ( bli_arch_isa_tier( arch_id ) ) \
    { \
        case BLIS_ISA_TIER_AVX512: \
            PASTECH(ZEN4_, ch) \
            /* Each ZEN4_{s,d,c,z} calls-and-returns only for bli_is_trans; \
               otherwise control falls through to the AVX2 case (same guard, \
               also false) and on to default (the dotxf path). */ \
        case BLIS_ISA_TIER_AVX2: \
            PASTECH(ZEN_, ch) \
        case BLIS_ISA_TIER_GENERIC: \
        default: \
        { \
            /* Non-zen / generic platforms use the dotxf loop. */ \
            if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
            ctype* x1; \
            ctype* y1; \
            ctype* A1; \
\
            PASTECH(ch,dotxf_ker_ft) dotxf_kr_ptr; \
            dim_t                    b_fuse; \
\
            dotxf_kr_ptr = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
            b_fuse       = bli_cntx_get_blksz_def_dt( dt, BLIS_DF, cntx ); \
\
            for ( i = 0; i < n0; i += f ) \
            { \
                f  = bli_determine_blocksize_dim_f( i, n0, b_fuse ); \
\
                A1 = a_buf + ( i * lda ) + ( 0 * inca ); \
                x1 = x_buf; \
                y1 = y_buf + ( i * incy ); \
\
                dotxf_kr_ptr \
                ( \
                    conja, conjx, m0, f, alpha, \
                    A1, inca, lda, \
                    x1, incx, \
                    beta, \
                    y1, incy, \
                    cntx \
                ); \
            } \
\
            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3); \
            return; \
        } \
    } \
\
}

INSERT_GENTFUNC_BASIC0( gemv_unf_var1 )
