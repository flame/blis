/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

/* In case the UKR Selectors are not defined when compiling
   the kernels(based on the ZEN architecture), we need to set them to empty values here */
#ifndef BLIS_FAMILY_AMDZEN
  #ifndef BLIS_FAMILY_ZEN
    #define ZEN_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel )
  #endif

  #ifndef BLIS_FAMILY_ZEN4
    #define ZEN4_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel )
  #endif

  #ifndef BLIS_FAMILY_ZEN5
    #define ZEN5_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel )
  #endif

  #ifndef BLIS_FAMILY_ZEN6
    #define ZEN6_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel )
  #endif
#endif

/* Defining the bli_?gemm_tiny interfaces */
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, tfuncname ) \
err_t PASTEMAC( ch, tfuncname ) \
    ( \
      trans_t transa, \
      trans_t transb, \
      dim_t  m0, \
      dim_t  n0, \
      dim_t  k0, \
      const ftype*    alpha, \
      const ftype*    a0, const inc_t rs_a0, const inc_t cs_a0, \
      const ftype*    b0, const inc_t rs_b0, const inc_t cs_b0, \
      const ftype*    beta, \
      ftype*    c0, const inc_t rs_c0, const inc_t cs_c0, \
      bool is_parallel \
    ) \
{ \
    /* Early return based on transpose values */ \
    if( transa == BLIS_CONJ_NO_TRANSPOSE || transa == BLIS_CONJ_TRANSPOSE || \
        transb == BLIS_CONJ_NO_TRANSPOSE || transb == BLIS_CONJ_TRANSPOSE ) \
        return BLIS_FAILURE; \
\
    /* Query the architecture ID */ \
    arch_t arch_id = bli_arch_query_id_internal(); \
    /* Declaring the object to hold the kernel information */ \
    gemmtiny_ukr_info_t gemmtiny_ukr_info; \
    /* Variable to flag success/failure of obtaining the kernel */ \
    err_t ukr_support = BLIS_NOT_YET_IMPLEMENTED; \
\
    /* Setting up the metadata for kernel acquisition */ \
    stor3_t stor_id = 0; \
    /* Local variables and pointers */ \
    ftype *a, *b, *c; \
    dim_t m, n, k; \
    inc_t rs_a, cs_a, rs_b, cs_b, rs_c, cs_c; \
\
    /* Assigning values to the local variables */ \
    a = (ftype *)a0; \
    b = (ftype *)b0; \
    c = (ftype *)c0; \
\
    m = m0; \
    n = n0; \
    k = k0; \
    /* Support for logical transpose of the operands */ \
    if( transa == BLIS_TRANSPOSE ) \
    { \
      rs_a = cs_a0; \
      cs_a = rs_a0; \
    } \
    else \
    { \
      rs_a = rs_a0; \
      cs_a = cs_a0; \
    } \
    if( transb == BLIS_TRANSPOSE ) \
    { \
      rs_b = cs_b0; \
      cs_b = rs_b0; \
    } \
    else \
    { \
      rs_b = rs_b0; \
      cs_b = cs_b0; \
    } \
    rs_c = rs_c0; \
    cs_c = cs_c0; \
\
    /* Generating the storage sequence, in the order C,A,B */ \
    stor_id = 4 * ( rs_c == 1 ) + \
              2 * ( rs_a == 1 ) + \
              1 * ( rs_b == 1 ); \
\
    /* Runtime acquisition of kernel based on the metadata and arch_ID */ \
    switch ( arch_id ) \
    { \
      case BLIS_ARCH_ZEN6: \
        ZEN6_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
      case BLIS_ARCH_ZEN5: \
        ZEN5_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
      case BLIS_ARCH_ZEN4: \
        ZEN4_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
      case BLIS_ARCH_ZEN3: \
      case BLIS_ARCH_ZEN2: \
      case BLIS_ARCH_ZEN: \
        ZEN_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
      default: \
        /* In case of other non-zen architectures, use an alternative path(functional) */ \
        /* This is done in order to avoid using context in this interface */ \
        return BLIS_FAILURE; \
    } \
    /* In case the storage sequence is not supported or thresholds are not met, return early */ \
    if( ukr_support == BLIS_NOT_YET_IMPLEMENTED ) \
      return BLIS_FAILURE; \
\
    /* Unwrapping the object details associated with the kernel */ \
    PASTECH2( ch, gemmsup, _ker_ft ) ukr_fp = ( PASTECH2( ch, gemmsup, _ker_ft ) )gemmtiny_ukr_info.ukr_fp; \
    bool ukr_pref = gemmtiny_ukr_info.stor_pref; \
    bool enable_pack = gemmtiny_ukr_info.enable_pack; \
    dim_t MR = gemmtiny_ukr_info.MR; \
    dim_t NR = gemmtiny_ukr_info.NR; \
    /* Setting a boolean to check for operation transpose(induce) */ \
    bool is_primary = !( ukr_pref ^ ( ( stor_id < 3 ) || ( stor_id == 4 ) ) ); \
\
    /* In case of inducing operation transpose, we need to alter the parameters */ \
    if( !is_primary ) \
    { \
      inc_t rs_at, cs_at, rs_bt, cs_bt; \
      rs_at = cs_b; \
      cs_at = rs_b; \
\
      rs_bt = cs_a; \
      cs_bt = rs_a; \
\
      rs_a = rs_at; \
      cs_a = cs_at; \
\
      rs_b = rs_bt; \
      cs_b = cs_bt; \
\
      rs_c = cs_c0; \
      cs_c = rs_c0; \
\
      a = (ftype *)b0; \
      b = (ftype *)a0; \
\
      m = n0; \
      n = m0; \
    } \
    /* Declaring the pointers and auxillary kernel info */ \
    /* Since we are primarily using the m-var SUP kernels, we need
       to set the panel stride for A matrix */ \
    /* The panel stride for A matrix would depend on whether we pack A or not */ \
    auxinfo_t aux; \
    ftype *a_panel = a; \
    ftype *b_panel = b; \
    ftype *c_panel = c; \
    inc_t ps_a_use, ps_b_use; \
\
    /* Pack the appropriate matrix, based on the toggle in the lookup table(i.e, enable_pack) */ \
    if( enable_pack == TRUE ) \
    { \
      /* Acquire the packing kernel */ \
      PASTECH2( ch, packm_cxk, _ker_ft ) pack_fp = ( PASTECH2( ch, packm_cxk, _ker_ft ) )gemmtiny_ukr_info. pack_fp; \
      /* The kernel's storage preference will suggest if we have to pack A or B matrix */ \
      /* This is because a row-stored kernel loads from B, while a column stored kernel loads from A */ \
      /* The metadata for the packed matrix should be changed accordingly */ \
      /* Declaring the variables/pointers for packing */ \
      ftype *input_buf, *pack_buf; \
      dim_t pack_size; \
      dim_t loop_iter, loop_left; \
      dim_t rs_input, cs_input, pack_dim; \
      /* Setting up a local variable to a scale factor of 1.0, to be passed to the packing kernel */ \
      ftype local_one; \
      PASTEMAC( ch, sets )( 1.0, 0.0, local_one ); \
      /* Declaring the necessary objects for acquiring memory from memory pool */ \
      mem_t mem_local; \
      rntm_t rntm_local; \
      if( ukr_pref == TRUE ) \
      { \
        dim_t n_pack = ( ( n + NR - 1 ) / NR ) * NR; \
        pack_size = n_pack * k; \
      } \
      else \
      { \
        dim_t m_pack = ( ( m + MR - 1 ) / MR ) * MR; \
        pack_size = m_pack * k; \
      } \
\
      /* Querying the pack-block allocator and attaching to the local runtime */ \
      bli_pba_rntm_set_pba( &rntm_local ); \
\
      /* Acquire the memory from the memory pool */ \
      bli_pba_acquire_m( &rntm_local, pack_size * sizeof( ftype ) , BLIS_BUFFER_FOR_A_BLOCK, &mem_local ); \
      ftype *buf_addr = bli_mem_buffer( &mem_local ); /* malloc( pack_size * sizeof( ftype ) ); */ \
\
      /* Exit if memory was not allocated */ \
      if( buf_addr == NULL )  return BLIS_FAILURE; \
      pack_buf = buf_addr; \
\
      /* Setting the metadata based on the ukr preference */ \
      if( ukr_pref == TRUE ) \
      { \
        /* Pack B matrix, since ukr is row-preferential */ \
        input_buf = b; \
        rs_input = cs_b; cs_input = rs_b; \
        loop_iter = n / NR; loop_left = n - ( loop_iter * NR ); \
        pack_dim = NR; \
        rs_b = NR; cs_b = 1; \
        ps_a_use = ( MR * rs_a ); \
        ps_b_use = ( NR * k ); \
        b_panel = buf_addr; \
      } \
      else \
      { \
        /* Pack A matrix, since ukr is col-preferential */ \
        input_buf = a; \
        rs_input = rs_a; cs_input = cs_a; \
        loop_iter = m / MR; loop_left = m - ( loop_iter * MR ); \
        pack_dim = MR; \
        rs_a = 1; cs_a = MR; \
        ps_a_use = ( MR * k ); \
        ps_b_use = ( NR * cs_b ); \
        a_panel = buf_addr; \
      } \
\
      /* Packing the appropriate matrix iteratively */ \
      for( dim_t i = 0; i < loop_iter; i += 1 ) \
      { \
        /* Call the packing kernel */ \
        pack_fp( BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS, pack_dim, k, k, &local_one, input_buf, rs_input, cs_input, pack_buf, pack_dim, NULL ); \
\
        /* Update the pointer for the next iteration */ \
        input_buf += pack_dim * rs_input; \
        pack_buf += pack_dim * k; \
      } \
      if( loop_left ) \
      { \
        /* Call the packing kernel */ \
        pack_fp( BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS, loop_left, k, k, &local_one, input_buf, rs_input, cs_input, pack_buf, pack_dim, NULL ); \
      } \
\
      /* Set the panel stride info for A matrix(since we use m-var kernels) */ \
      bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
      /* Setting up the variables for blocked iterations */ \
      /* The m-var SUP kernels operate on A(m x k), B(k x NR)
          and C(m x NR). Thus, we need to block the data in the
          n-dimension before calling the kernel(that is, the NR loop) */ \
      dim_t n_iter = n / NR; \
      dim_t n_rem = n - ( n_iter * NR ); \
      dim_t j = 0; \
      /* Operating on the main-case of n(NR) */ \
      for( ; j < n_iter; j += 1 ) \
      { \
        ukr_fp \
        ( \
          BLIS_NO_CONJUGATE, \
          BLIS_NO_CONJUGATE, \
          m, \
          NR, \
          k, \
          (ftype* restrict)alpha, \
          (ftype* restrict)a_panel, rs_a, cs_a, \
          (ftype* restrict)b_panel, rs_b, cs_b, \
          (ftype* restrict)beta, \
          (ftype* restrict)c_panel, rs_c, cs_c, \
          &aux, \
          NULL \
        ); \
        b_panel += ps_b_use; \
        c_panel += NR * cs_c; \
      } \
      /* Operating on the fringe case of n(<NR) */ \
      if( n_rem ) \
      { \
        ukr_fp \
        ( \
          BLIS_NO_CONJUGATE, \
          BLIS_NO_CONJUGATE, \
          m, \
          n_rem, \
          k, \
          (ftype* restrict)alpha, \
          (ftype* restrict)a_panel, rs_a, cs_a, \
          (ftype* restrict)b_panel, rs_b, cs_b, \
          (ftype* restrict)beta, \
          (ftype* restrict)c_panel, rs_c, cs_c, \
          &aux, \
          NULL \
        ); \
      } \
\
      /* Release the memory back to the pool */ \
      bli_pba_release( &rntm_local, &mem_local ); \
      /* free( buf_addr ); */ \
    } \
    else \
    { \
      /* Set the panel stride info for A matrix(since we use m-var kernels) */ \
      ps_a_use = ( MR * rs_a ); \
      ps_b_use = ( NR * cs_b ); \
      bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
      /* Setting up the variables for blocked iterations */ \
      /* The m-var SUP kernels operate on A(m x k), B(k x NR)
          and C(m x NR). Thus, we need to block the data in the
          n-dimension before calling the kernel(that is, the NR loop) */ \
      dim_t n_iter = n / NR; \
      dim_t n_rem = n - ( n_iter * NR ); \
      dim_t j = 0; \
      /* Operating on the main-case of n(NR) */ \
      for( ; j < n_iter; j += 1 ) \
      { \
        ukr_fp \
        ( \
          BLIS_NO_CONJUGATE, \
          BLIS_NO_CONJUGATE, \
          m, \
          NR, \
          k, \
          (ftype* restrict)alpha, \
          (ftype* restrict)a_panel, rs_a, cs_a, \
          (ftype* restrict)b_panel, rs_b, cs_b, \
          (ftype* restrict)beta, \
          (ftype* restrict)c_panel, rs_c, cs_c, \
          &aux, \
          NULL \
        ); \
        b_panel += ps_b_use; \
        c_panel += NR * cs_c; \
      } \
      /* Operating on the fringe case of n(<NR) */ \
      if( n_rem ) \
      { \
        ukr_fp \
        ( \
          BLIS_NO_CONJUGATE, \
          BLIS_NO_CONJUGATE, \
          m, \
          n_rem, \
          k, \
          (ftype* restrict)alpha, \
          (ftype* restrict)a_panel, rs_a, cs_a, \
          (ftype* restrict)b_panel, rs_b, cs_b, \
          (ftype* restrict)beta, \
          (ftype* restrict)c_panel, rs_c, cs_c, \
          &aux, \
          NULL \
        ); \
      } \
    } \
\
    return BLIS_SUCCESS; \
} \

GENTFUNC(    float, s, gemm_tiny )

/*
 * bli_gemm_tiny_single_threaded:
 *
 * Datatype-generic decision of whether a tiny GEMM of size (M × N × K) should
 * run single-threaded (ST) or multi-threaded (MT). Shared by s/d/c/z tiny
 * paths - the caller passes the kernel's MR/NR, the per-core GFLOP rate, the
 * threading overhead and the datatype (which selects 2 FLOPs/MAC for real or
 * 8 for complex).
 *
 * Arguments:
 *   M, N, K    : matrix dimensions
 *   MR, NR     : Microtile of kernel
 *   GF_core    : performance per core in GFLOPs
 *   T_over_us  : per-call threading overhead in microseconds
 *   dt         : datatype (real -> 2 FLOPs/MAC, complex -> 8 FLOPs/MAC)
 *
 * Method:
 *   - The model expects MR and NR of microkernel.
 *   - Each microkernel tile performs:
 *         FLOPs_per_K_iter = flops_per_mac * MR * NR  (e.g. 2*24*8 = 384 for d)
 *   - The function computes a K-threshold:
 *
 *         base = (T_over_s * GF_core * 1e9) / FLOPs_per_K_iter
 *         K_thresh = (alpha * base) / tiles_per_thread
 *
 *     where alpha (≈0.3) is an amortization factor that accounts
 *     for partial overlap of compute and overhead. It can be tuned
 *     as per requirement.
 *
 * Decision:
 *   - If K ≥ K_thresh → problem is "long" enough in the K-loop,
 *     so overhead is amortized → MT is beneficial.
 *   - If K < K_thresh → inner loop too short, overhead dominates,
 *     so ST is safer.
 *
 * Example:
 *   Input:  M=114, N=114, K=45, GF_core=50 GF/s, T_over=15 µs
 *
 *   tiles_M = ceil(114/24) = 5
 *   tiles_N = ceil(114/8)  = 15
 *   total_tiles = 5 × 15 = 75
 *   tiles_per_thread ≈ 38
 *
 *   FLOPs_per_K = 384
 *   base = (15e-6 * 50e9) / 384 ≈ 1953
 *   K_thresh = 0.3 × 1953 / 38 ≈ 15.4
 *
 *   Since K=45 > K_thresh=15.4, → MT chosen.
 *
 * Return:
 *   - true  → prefer single-thread
 *   - false → prefer multi-thread
 */

static bool bli_gemm_tiny_single_threaded(dim_t M, dim_t N, dim_t K, dim_t MR, dim_t NR, double GF_core, double T_over_us, num_t dt)
{
    const double alpha = 0.3; // amortization factor
    double K_thresh = 0; 

    // Convert thread overhead to seconds
    double T_over_s = T_over_us * 1e-6;

    // Compute tiles in M and N
    dim_t tiles_M = (M + MR - 1) / MR;
    dim_t tiles_N = (N + NR - 1) / NR;

	dim_t thread_num = 2;/* Considering 2 threads */
    // Tiles assigned to each thread (assuming splitting along M)
    dim_t tiles_per_thread = ( ( (tiles_M + thread_num - 1 ) / thread_num ) * tiles_N );

    // FLOPs per multiply-accumulate: 2 for real (s/d), 8 for complex (c/z).
    // GF_core must be expressed in the matching (real or complex) FLOP rate.
    dim_t flops_per_mac = ( dt == BLIS_SCOMPLEX || dt == BLIS_DCOMPLEX ) ? 8 : 2;

    // Compute K threshold
	dim_t FLOPS_per_k_iter = ((MR * NR) * flops_per_mac);
	// base depends on parameters T_over_s and GF_core and FLOPS_per_k_iter.
	// Where FLOPS_per_k_iter is computed based on flops per cycle and micro-kernel shape.
	// T_over_s is thread overhead time(time of forking, barrier, joining).
	// GF_core is calculated based on flops per cycle * clock frequency of CPU. Currently it is
	// considered as 16 flops * 3.7GHz.
	// Note: It may change from platform to platform and even may vary among various threading library
	// implementations, versions.
    double base = (T_over_s * GF_core * 1e9) / FLOPS_per_k_iter;
    K_thresh = alpha * base / (double)tiles_per_thread;

    return (K >= K_thresh) ? false : true;
}

err_t bli_dgemm_tiny
(
        trans_t transa,
        trans_t transb,
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const double*    alpha,
        const double*    a, const inc_t rs_a0, const inc_t cs_a0,
        const double*    b, const inc_t rs_b0, const inc_t cs_b0,
        const double*    beta,
        double*    c, const inc_t rs_c0, const inc_t cs_c0
)
{
    // Query the architecture ID
    arch_t arch_id = bli_arch_query_id_internal();
    bool is_mt = bli_thread_get_is_parallel();
    {
        // Dispatch on ISA capability tier rather than on individual arch_id
        // values. bli_arch_isa_tier() is the single source of truth mapping a
        // Zen arch to its tier, so new Zen parts only need to be added there.
        switch ( bli_arch_isa_tier( arch_id ) )
        {
          case BLIS_ISA_TIER_AVX512:
          // Compile-time guard: the AVX-512 tiny kernel is only built when the
          // Zen4 kernel set is present. Without it this reference would fail to
          // link in configs that don't build those kernels (the runtime tier
          // check alone does not guarantee the symbol exists in the binary).
#if defined(BLIS_KERNELS_ZEN4)
              if(((m == n) && (m < 400) && (k < 1000)) ||
              ( (m != n) && (( ((m + n -k) < 1500) &&
              ((m + k-n) < 1500) && ((n + k-m) < 1500) ) ||
              /**
			   *  Since tiny gemm interface passes the entire input as is to the micro-kernel itself,
			   *  which loops as jr, ir and k.  For large M dimension it will keep flushing B matrix out
			   *  of cache while loading new panel of A and C matrix, and it may again need to reload previously loaded
			   *  B panel. Such repetitive data movement for B matrix hampers the performance. So limiting the inputs
			   *  for tiny gemm interface for optimal performance.
			   */
              ((m < 10000) && (n <= 100) && (k <=100)))))
              {
                  if( (is_mt == FALSE) ||
                  ( bli_gemm_tiny_single_threaded(m, n, k, 24/*MR*/, 8/*NR*/, 60/*Core's Gflops*/, 15/*Threading_overhead*/, BLIS_DOUBLE) == TRUE ) )
                  {
                      /* single threaded execution */
                      return bli_dgemm_tiny_zen4_24x8
                      (
                       1 * (transa == BLIS_CONJ_NO_TRANSPOSE),
                       1 * (transb == BLIS_CONJ_NO_TRANSPOSE),
                       transa,
					   transb,
					   m,
					   n,
					   k,
					   alpha,
					   a, rs_a0, cs_a0,
					   b, rs_b0, cs_b0,
					   beta,
					   c, rs_c0, cs_c0
                      );
                  }
              }
#endif
              break;

          case BLIS_ISA_TIER_AVX2:
              if( is_mt == FALSE )
              {
                  if( ( (m <= 8)  || ( (m <= 1000) && (n <= 24) && (k >= 4) ) ) && (k <= 1500) )
                  {
                  return bli_dgemm_tiny_zen_6x8
                          (
                              1 * (transa == BLIS_CONJ_NO_TRANSPOSE),
                              1 * (transb == BLIS_CONJ_NO_TRANSPOSE),
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              a, rs_a0, cs_a0,
                              b, rs_b0, cs_b0,
                              beta,
                              c, rs_c0, cs_c0
                          );
                  }
              }
              break;
          default:
              return BLIS_FAILURE;
        }
    }

    return BLIS_FAILURE;
}

err_t bli_zgemm_tiny
    (
      trans_t transa,
      trans_t transb,
      dim_t  m,
      dim_t  n,
      dim_t  k,
      const dcomplex*    alpha,
      const dcomplex*    a, const inc_t rs_a0, const inc_t cs_a0,
      const dcomplex*    b, const inc_t rs_b0, const inc_t cs_b0,
      const dcomplex*    beta,
      dcomplex*    c, const inc_t rs_c0, const inc_t cs_c0,
      bool is_parallel
    )
{
    // Query the architecture ID
    arch_t arch_id = bli_arch_query_id_internal();
    bool is_mt = is_parallel;
    // Dispatch on ISA capability tier rather than on individual arch_id
    // values. bli_arch_isa_tier() is the single source of truth mapping a
    // Zen arch to its tier, so new Zen parts only need to be added there.
    switch ( bli_arch_isa_tier( arch_id ) )
    {
        case BLIS_ISA_TIER_AVX512:
        // Compile-time guard: the AVX-512 tiny kernel is only built when the
        // Zen4 kernel set is present. Without it this reference would fail to
        // link in configs that don't build those kernels (the runtime tier
        // check alone does not guarantee the symbol exists in the binary).
#if defined(BLIS_KERNELS_ZEN4)
       /**
        * Note: Kernel supports the following combinations of Op(A) and Op(B):
        *   - Op(A) = A,       Op(B) = B        (NO_TRANSPOSE, NO_TRANSPOSE)
        *   - Op(A) = A^T,     Op(B) = B        (TRANSPOSE, NO_TRANSPOSE)
        *   - Op(A) = A,       Op(B) = B^T      (NO_TRANSPOSE, TRANSPOSE)
        *   - Op(A) = A^T,     Op(B) = B^T      (TRANSPOSE, TRANSPOSE)
        *   - Op(A) = conj(A), Op(B) = B        (CONJ_NO_TRANSPOSE, NO_TRANSPOSE)
        *   - Op(A) = A^H,     Op(B) = B        (CONJ_TRANSPOSE, NO_TRANSPOSE)
        *   - Op(A) = A,       Op(B) = conj(B)  (NO_TRANSPOSE, CONJ_NO_TRANSPOSE)
        *   - Op(A) = A,       Op(B) = B^H      (NO_TRANSPOSE, CONJ_TRANSPOSE)
        *   - Op(A) = A^T,     Op(B) = conj(B)  (TRANSPOSE, CONJ_NO_TRANSPOSE)
        *   - Op(A) = A^T,     Op(B) = B^H      (TRANSPOSE, CONJ_TRANSPOSE)
        *   - Op(A) = conj(A), Op(B) = B^T      (CONJ_NO_TRANSPOSE, TRANSPOSE)
        *   - Op(A) = A^H,     Op(B) = B^T      (CONJ_TRANSPOSE, TRANSPOSE)
        *
        * However framework changes are needed for:
        *   - Op(A) = conj(A), Op(B) = conj(B)  (CONJ_NO_TRANSPOSE, CONJ_NO_TRANSPOSE)
        *   - Op(A) = A^H,     Op(B) = B^H      (CONJ_TRANSPOSE, CONJ_TRANSPOSE)
        *   - Op(A) = conj(A), Op(B) = B^H      (CONJ_TRANSPOSE, CONJ_NO_TRANSPOSE)
        *   - Op(A) = A^H,     Op(B) = conj(B)  (CONJ_NO_TRANSPOSE, CONJ_TRANSPOSE)
        * So currently these remain unsupported for zen4/zen5.
        * TODO: add framework support for these combinations.
        */
        if( ( m < 300 ) && ( n < 300 ) && ( k < 300 ) &&
            !( bli_does_conj( transa ) && bli_does_conj( transb ) )
          )
        {
            if(is_mt == FALSE)
            {
                /* single threaded execution */
                return bli_zgemm_tiny_zen4_12x4
                (
                    ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a0, cs_a0,
                    b, rs_b0, cs_b0,
                    beta,
                    c, rs_c0, cs_c0
                );
            }
        }
#endif
        break;

        case BLIS_ISA_TIER_AVX2:
        if( is_mt == FALSE )
        {
        /**
         * Note conjugate A, B matrices are not supported for zen/2/3.
         */
            if( ( bli_is_notrans( transa ) && ( m < 60 ) && ( n >= 4 ) && ( n < 200 ) && ( k < 68 ) && (m % 2 == 0) ) ||
            ( bli_is_trans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 16 ) && (m % 2 == 0) ) )
            {
                return bli_zgemm_tiny_zen_3x4
                (
                    ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a0, cs_a0,
                    b, rs_b0, cs_b0,
                    beta,
                    c, rs_c0, cs_c0
                );
            }
        }
        break;

        default:
            return BLIS_FAILURE;
    }

    return BLIS_FAILURE;

}

err_t bli_cgemm_tiny
    (
      trans_t transa,
      trans_t transb,
      dim_t  m,
      dim_t  n,
      dim_t  k,
      const scomplex*    alpha,
      const scomplex*    a, const inc_t rs_a0, const inc_t cs_a0,
      const scomplex*    b, const inc_t rs_b0, const inc_t cs_b0,
      const scomplex*    beta,
      scomplex*    c, const inc_t rs_c0, const inc_t cs_c0,
      bool is_parallel
    )
{
    // Query the architecture ID
    arch_t arch_id = bli_arch_query_id_internal();
    bool is_mt = is_parallel;
    // Dispatch on ISA capability tier rather than on individual arch_id
    // values. bli_arch_isa_tier() is the single source of truth mapping a
    // Zen arch to its tier, so new Zen parts only need to be added there.
    switch ( bli_arch_isa_tier( arch_id ) )
    {
        case BLIS_ISA_TIER_AVX512:
        // Compile-time guard: the AVX-512 tiny kernel is only built when the
        // Zen4 kernel set is present. Without it this reference would fail to
        // link in configs that don't build those kernels (the runtime tier
        // check alone does not guarantee the symbol exists in the binary).
#if defined(BLIS_KERNELS_ZEN4)
       /**
        * Note: Kernel supports ALL 16 combinations of Op(A) and Op(B) for
        *       trans_t in {NO_TRANSPOSE, TRANSPOSE, CONJ_NO_TRANSPOSE,
        *       CONJ_TRANSPOSE}. The conja/conjb flags forwarded to the
        *       24x4 micro-kernel below select among the 4 conj-aware
        *       MICRO_TILE variants (NN, CONJA, CONJB, CONJA_CONJB) which
        *       are all implemented in bli_gemmsup_cv_zen4_asm_c24x4m.c.
        */
        if( ( m < 300 ) && ( n < 300 ) && ( k < 300 ) )
        {
            /* Route to the single-threaded tiny cgemm kernel when the caller did
             * not request parallelism, or when the problem is too small to
             * amortize the OpenMP team launch/barrier (e.g. 10x10x10, where 16
             * threads ran ~4x slower than one core).
             *
             * GF_core is the per-core cgemm throughput = flops/cycle * clock. A
             * 512-bit register holds 16 single-precision float lanes and the
             * AVX-512 FMA units sustain ~32 flops/cycle, giving
             * 32 flops/cycle * 3.7 GHz ~= 118 GFLOP/s. The 24x4 register tile is
             * that of bli_cgemm_tiny_zen4_24x4 dispatched below. */
            if( (is_mt == FALSE) ||
                ( bli_gemm_tiny_single_threaded(m, n, k, 24/*MR*/, 4/*NR*/, 118/*Core's Gflops*/, 15/*Threading_overhead*/, BLIS_SCOMPLEX) == TRUE ) )
            {
                /* single threaded execution */
                return bli_cgemm_tiny_zen4_24x4
                (
                    ((transa == BLIS_CONJ_NO_TRANSPOSE) || (transa == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    ((transb == BLIS_CONJ_NO_TRANSPOSE) || (transb == BLIS_CONJ_TRANSPOSE)) ? BLIS_CONJUGATE : BLIS_NO_CONJUGATE,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    a, rs_a0, cs_a0,
                    b, rs_b0, cs_b0,
                    beta,
                    c, rs_c0, cs_c0
                );
            }
        }
#endif
        break;

        default:
            return BLIS_FAILURE;
    }

    return BLIS_FAILURE;

}

bool bli_is_sgemm_tiny_zen
  (
    stor3_t stor_id, 
    trans_t transa, 
    trans_t transb, 
    dim_t m, 
    dim_t n, 
    dim_t k, 
    bool is_parallel, 
    dim_t NR, 
    dim_t NUM_FLOATS_IN_CACHE_LINE, 
    dim_t NUM_FLOATS_IN_L1, 
    dim_t NUM_FLOATS_IN_L2
  )
{
  dim_t elems_in_L1 = 0;
  dim_t elems_in_L2 = 0;

  if (m > 682 || n > 512 || k > 240)
  {
    // Don't enter tiny path if any dimension exceeds its SUP threshold
    // The thresholds are derived from the sizes that enter the SUP path
    // Note: These thresholds are conservative because in SUP path,
    // matrices will enter SUP if any of the above thresholds are satisfied
    // regardless of the size of the other dimensions
    return false;
  }

  // Stores total number of elements in matrices A, B and C (which needs to at least fit in the L2 cache)
  elems_in_L2 = ((m + k) * n)  + (m * k); 

  if(elems_in_L2 > NUM_FLOATS_IN_L2)
  {
    // For large sizes, we dont go into the tiny path at all
    return false;
  }

  if(stor_id == BLIS_CRC || stor_id == BLIS_CCR || stor_id == BLIS_CCC || stor_id == BLIS_RCC)
  {
    // all these cases have an induced transpose, so we need to swap the m and n values
    // this is needed because the analysis that follows later is based on these dimensions
    dim_t tmp = m;
    m = n;
    n = tmp;
  }
  
  if(stor_id != BLIS_RRC && stor_id != BLIS_CRC)
  {
    // these are the cases which call the rv-m kernels
    // The access pattern for these kernels is as follows:
    //     Matrix A (M x K):               Matrix B (K x NR):              Matrix C (M x NR):
    // +-------------+                +--------------------------+   +--------------------------+
    // | * -> * -> * | row 0          | ->->->->->->->-> (64 col)|   | *  *  *  * ... (64 cols) |
    // | * -> * -> * | row 1          | ->->->->->->->->         |   | *  *  *  *               |
    // | * -> * -> * | row 2          | ->->->->->->->->         |   | *  *  *  *               |
    // | * -> * -> * | row 3          | ->->->->->->->->         |   | *  *  *  *               |
    // | * -> * -> * | row 4          |      ...                 |   | *  *  *  *               |
    // | * -> * -> * | row 5          +--------------------------+   | *  *  *  *               |
    // |      ...    |                  K rows (4 per iter)          |      ...                 |
    // +-------------+                  Each row = 64 elements        +--------------------------+
    //   6 rows                                                         6 rows x 64 cols
    //   (broadcast scalars)            (vector loads)
    // We broadcast 6 elements from A and load NR elements from B
    // In the K-loop we loop over the columns of A and rows of B.
    // Once the K-loop is complete, we iterate over the full M dimension of C and A
    // The data-reuse pattern forms the basis of the decision to enter the tiny-path
    //
    // For the A-matrix, we want the entire A-matrix to stay in the L1 cache,
    // This is a conservative approach, mainly because the SUP path chooses rv-n or rv-m 
    // kernels in certain cases. i.e we loop over the M or N dimension of A 
    // depending on which kernel is chosen. This is the most conservative approach to select
    // this path where we are always at least as fast as the SUP path. Also in the case
    // of the rv-m kernel (which is used in the tiny path), the outer loop (which calls the kernels)
    // loops over NR. Thus we reuse the entire A matrix in the outer loop and it 
    // makes sense to keep the full matrix in L1. If the SUP path removes the rv-n 
    // kernels at a later date, we could possibly fine tune this further.
    elems_in_L1 = k * m; 
    
    // For the B-matrix, this is fully re-used inside the rv-m kernel
    // so we keep this panel in L1 => k * min(n, NR)
    elems_in_L1 += k * bli_min(n, NR);
    
    // For the C-matrix, we loop over MRxNR size at a time while the outer loop repeats this till we cover the full M dimension
    // when C-matrix has the Col storage scheme, we want to make sure that we have enough capacity
    // in the L1 cache so that all the cache-lines that are loaded when accessing one micro-kernel 
    // stays in the cache, this is min(NR,n) * min(CACHE_LINE_SIZE, m) 
    // remember that the C-matrix has no-reuse so we dont need to keep a larger chunk than this in memory
    // we have a min operator to take care of cases where n/m are lesser than the NR or cache-line-size respectively 
    // The C-matrix analysis needs to be modified slightly for the cases where fringe kernels
    // are used. For the rv-m kernels (NR==64), the fringe kernel sizes are 48,32,16,8,4,2,1
    // Consider the example where the 4 and 2 fringe kernels are executed. In this case
    // we load > 4 elements on a single access of C (since cache line size is 64B) but 
    // if the M dimension is long enough, these memory elements are no longer in the cache
    // when we eventually execute the 2-fringe kernel. In the SUP path, this is not such
    // a problem because we have blocking in the MC direction such that we finish all the 
    // fringe kernels in the MC block before moving on to the next MC loop.
    // But in the tiny path, we loop over the entire M dimension so this has to be accounted for.
    // Thus we need to make space for at least 8+4+2+1=15 fringe kernels in the worst case
    // for the larger fringe kernel transitions, the amount of computation offsets the 
    // memory loading and this effect is not so pronounced for these cases
    // thus we need to make extra room for these fringe cases in the worst case
    // i.e where we access the 8,4,2,1 kernels in that order

    // for a given n, we compute the worst case fringe size, this is not 
    // the exact size (based on the fringe kernel calls) but gives an upper-bound
    dim_t fringe_size = m * bli_min(15, n % NR);
    elems_in_L1 += bli_max(fringe_size, bli_min(NR, n) * bli_min(NUM_FLOATS_IN_CACHE_LINE, m));
  }
  else
  {
    // these are the cases which call the rd-m kernels
    // The access pattern for these kernels is as follows:
    //     Matrix A (M x K):               Matrix B (K x NR):              Matrix C (M x NR):
    // +-------------+                +--------------------------+   +--------------------------+
    // | ->->->->->  | row 0          | |  |  |  | ... (64 cols)|   | *  *  *  * ... (64 cols) |
    // | ->->->->->  | row 1          | |  |  |  |              |   | *  *  *  *               |
    // | ->->->->->  | row 2          | |  |  |  |              |   | *  *  *  *               |
    // | ->->->->->  | row 3          | |  |  |  |              |   | *  *  *  *               |
    // | ->->->->->  | row 4          | |  |  |  |              |   | *  *  *  *               |
    // | ->->->->->  | row 5          | |  |  |  |              |   | *  *  *  *               |
    // |      ...    |                |      ...                |   |      ...                 |
    // +-------------+                +--------------------------+   +--------------------------+
    //   6 rows                         K rows x 64 cols              6 rows x 64 cols
    //   (processes K elements/row)
    // In one k-iteration of the kernel, we load 6 rows of A and 4 columns of B
    // note that this one operation only produces a small 6x4 tilelet of C
    // This kernel then loops over the M-dimension to produce the Mx4 panel of C
    // In the outer kernel loop we loop over the N-direction of C and B to finally
    // produce the MxNR panel of C. This is the case for the rd-m kernel which 
    // loops over the M-direction inside the kernel

    // Again we need to follow a conservative approach for modelling these accesses
    // this is again caused by the fact that the SUP path uses rd-m and rd-n for 
    // the RRC and CRC cases respectively
    // if the SUP-path only used rd-m kernels, the analysis would have been as follows:
    // for the A matrix, since we reuse the full matrix inside the kernel, we need to 
    // store the full matrix in L1 => m * k
    // For the B matrix, we only need to store one panel just so that 
    // it doesn't remove any of the other matrix elements from L1 => 4*k
    // For the C-matrix, we would have had make sure that any of the
    // matrix cache line reads dont go out of the L1 cache before the
    // next N-dimension iteration => m * min (CACHE_LINE_SIZE, n)
    // But for the rd-n kernels, we have a slightly different analysis
    // Here we reuse the entire B-matrix in every loop inside the kernel
    // so for the B-matrix we need =>  n * k
    // We only need to allocate one row panel for A => 4 * k
    // for the C matrix, in order not to waste CACHE_LINE loads
    // we need => n * min(m, MR) 
    // remember, we loop over the N-direction inside the kernel for the rd-n case
    
    // thus, in order to have a conservative modeling, we have the following conditions,
    // These can be relaxed at a later date if the SUP path either chooses the m or n
    // kernel variants appropriately. Alternatively, we could set these cases on a 
    // per-storage format basis, where we are sure which kernel is used in the SUP path
    // but not doing this now, in case we remove some of the variants in the SUP path

    // Note we dont have to handle fringe cases like the rv kernel case because
    // the modelling done here is very conservative at the moment
    elems_in_L1 = m * n;
    elems_in_L1 += bli_max(m, n) * k;
    elems_in_L1 += 4 * k;
  }

  if(elems_in_L1 < NUM_FLOATS_IN_L1)
  {
    return true;
  }
  else
  {
    return false;
  }
}
