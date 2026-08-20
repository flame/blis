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

#ifdef BLIS_ENABLE_OPENMP
#include <omp.h>

  #define INIT_NT(NT) dim_t NT = 1;
  // if openmp enabled, spawn threads.
  #define SPAWN_THREADS    _Pragma("omp parallel num_threads(NT)")

  // helper macros
  #define GET_TID()        omp_get_thread_num()
  #define GET_NT_REAL()    omp_get_num_threads()

  // helper macro for partitioning work.
  #define PARTITION_WORK(N, thread_start, job_per_thread) \
          bli_thread_vector_partition( N, GET_NT_REAL(), &thread_start, &job_per_thread, GET_TID() );

  // Determine the ideal number of threads. This reuses gemv's L2 thread
  // heuristic so that ger's work distribution matches gemv. bli_nthreads_l2
  // internally honours AOCL_DYNAMIC when it is enabled and otherwise falls
  // back to the number of threads requested by the user, so this is called
  // unconditionally (matching gemv) rather than being gated on AOCL_DYNAMIC.
  #define GET_DYNAMIC_N_THREADS(M, N, NT, ch) \
    bli_nthreads_l2                           \
    (                                         \
        BLIS_GEMV_KER,                        \
        PASTEMAC(ch,type),                    \
        BLIS_NO_TRANSPOSE,                    \
        bli_arch_query_id_internal(),         \
        M,                                    \
        N,                                    \
        &NT                                   \
    );
#else

  // place holder macros if openmp is not enabled. Threading (including the
  // NT thread-count query) is only meaningful under OpenMP, so NT is neither
  // declared nor referenced here -- this avoids an undeclared-NT compile
  // break for non-OpenMP (e.g. pthreads) builds with AOCL_DYNAMIC enabled.
  #define INIT_NT(NT)
  #define SPAWN_THREADS
  #define PARTITION_WORK(N, thread_start, job_per_thread)
  #define GET_DYNAMIC_N_THREADS(M, N, NT, ch)
#endif

#if defined(BLIS_KERNELS_ZEN4)

  // in order to keep the work distribution same between gemv and ger, gemv thresholds
  // are also used for ger.
  #define ZEN4_SHOULD_USE_ST_L2(M, N, dt)  bli_gemvst_thresh_is_met_zen4(M, N, BLIS_NO_TRANSPOSE, dt)
#else
  #define ZEN4_SHOULD_USE_ST_L2(M, N, dt)  true
#endif

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3) \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a1t; \
	ctype*  chi1; \
	ctype*  y1; \
	ctype   alpha_chi1; \
	dim_t   i; \
\
	PASTECH(ch,axpyv_ker_ft) kfp_av; \
\
  /* Query the context for the kernel function pointer. */ \
  kfp_av = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
  /* Set default to single threaded code. */ \
  bool is_st = true; \
  \
  /* If other L2 APIs are parallel for current arch, then parallelize GER */ \
  switch ( bli_arch_query_id_internal() ) \
  { \
  case BLIS_ARCH_ZEN6: \
  case BLIS_ARCH_ZEN5: \
  case BLIS_ARCH_ZEN4: \
    /* Currently only AVX512 is fully parallel for all GEMV. */ \
    /* Use multiple threads only if GEMV would use MT for same size */ \
    /* to ensure less probability of cache misses if multiple L2 APIs*/ \
    /* with same inputs are invoked by the application. */ \
    is_st = ZEN4_SHOULD_USE_ST_L2(m, n, dt); \
    break; \
\
  default: \
    break; \
  } \
\
  /* Avoid overhead of omp for small sizes.*/ \
  if ( is_st ) \
  { \
    for ( i = 0; i < m; ++i ) \
    { \
      a1t  = a + (i  )*rs_a + (0  )*cs_a; \
      chi1 = x + (i  )*incx; \
      y1   = y + (0  )*incy; \
\
      /* a1t = a1t + alpha * chi1 * y; */ \
      PASTEMAC(ch,copycjs)( conjx, *chi1, alpha_chi1 ); \
      PASTEMAC(ch,scals)( *alpha, alpha_chi1 ); \
\
      kfp_av \
      ( \
        conjy, \
        n, \
        &alpha_chi1, \
        y1,  incy, \
        a1t, cs_a, \
        cntx  \
      ); \
    } \
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
    return; \
  } \
\
  /* Set num threads to 1 by default. */ \
  INIT_NT(NT); \
\
  /* if AOCL_DYNAMIC is enabled, set nt to ideal number of threads.*/ \
  GET_DYNAMIC_N_THREADS(m, n, NT, ch); \
\
  /* Spawn nt number of threads if openmp is enabled, else don't spawn any threads */ \
  SPAWN_THREADS \
  { \
    /* Thread local variables*/ \
    ctype*  a1_tl; \
    ctype*  y1_tl; \
    ctype*  chi1_tl; \
    ctype   alpha_chi1_tl; \
    dim_t   i_tl; \
\
    dim_t job_per_thread = m; \
    dim_t thread_start   = 0; \
    /* Partition work along M dimension if openmp is enabled*/ \
    PARTITION_WORK(m, thread_start, job_per_thread); \
    for ( i_tl = thread_start; i_tl < thread_start + job_per_thread; ++i_tl ) \
    { \
      a1_tl   = a + (i_tl)*rs_a + (0  )*cs_a; \
      chi1_tl = x + (i_tl)*incx; \
      y1_tl   = y + (0   )*incy; \
  \
      /* a1t = a1t + alpha * chi1 * y; */ \
      PASTEMAC(ch,copycjs)( conjx, *chi1_tl, alpha_chi1_tl ); \
      PASTEMAC(ch,scals)( *alpha, alpha_chi1_tl ); \
  \
      kfp_av \
      ( \
        conjy, \
        n, \
        &alpha_chi1_tl, \
        y1_tl,  incy, \
        a1_tl, cs_a, \
        cntx  \
      ); \
    } \
  } \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
\
}

INSERT_GENTFUNC_BASIC0( ger_unb_var1 )

