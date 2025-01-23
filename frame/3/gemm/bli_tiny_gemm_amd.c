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
    arch_t arch_id = bli_arch_query_id(); \
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
    /* Setting up the auxillary kernel info */ \
    /* Since we are primarily using the m-var SUP kernels, we need
       to set the panel stride for A matrix */ \
    auxinfo_t aux; \
    inc_t ps_a_use = ( MR * rs_a ); \
    bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
    /* Setting up the variables for blocked iterations */ \
    /* The m-var SUP kernels operate on A(m x k), B(k x NR)
       and C(m x NR). Thus, we need to block the data in the
       n-dimension before calling the kernel(that is, the NR loop) */ \
    dim_t n_iter = n / NR; \
    dim_t n_rem = n - ( n_iter * NR ); \
    dim_t j = 0; \
    ftype *b_panel, *c_panel; \
    /* Operating on the main-case of n(NR) */ \
    for( ; j < n_iter; j += 1 ) \
    { \
      b_panel = b + ( j * NR * cs_b ); \
      c_panel = c + ( j * NR * cs_c ); \
      ukr_fp \
      ( \
        BLIS_NO_CONJUGATE, \
        BLIS_NO_CONJUGATE, \
        m, \
        NR, \
        k, \
        (ftype* restrict)alpha, \
        (ftype* restrict)a, rs_a, cs_a, \
        (ftype* restrict)b_panel, rs_b, cs_b, \
        (ftype* restrict)beta, \
        (ftype* restrict)c_panel, rs_c, cs_c, \
        &aux, \
        NULL \
      ); \
    } \
    /* Operating on the fringe case of n(<NR) */ \
    if( n_rem ) \
    { \
      b_panel = b + ( j * NR * cs_b ); \
      c_panel = c + ( j * NR * cs_c ); \
      ukr_fp \
      ( \
        BLIS_NO_CONJUGATE, \
        BLIS_NO_CONJUGATE, \
        m, \
        n_rem, \
        k, \
        (ftype* restrict)alpha, \
        (ftype* restrict)a, rs_a, cs_a, \
        (ftype* restrict)b_panel, rs_b, cs_b, \
        (ftype* restrict)beta, \
        (ftype* restrict)c_panel, rs_c, cs_c, \
        &aux, \
        NULL \
      ); \
    } \
    return BLIS_SUCCESS; \
} \

GENTFUNC( dcomplex, z, gemm_tiny )

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
    arch_t arch_id = bli_arch_query_id();
    //for the below tiny sizes of matrix, we force it to be ST compute.
    if(m <= 24 && n <= 24 && k <= 20)
    {
        switch (arch_id)
        {
          case BLIS_ARCH_ZEN5:
	  case BLIS_ARCH_ZEN4:
#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)
		return bli_dgemm_tiny_24x8
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
#endif
		break;

          case BLIS_ARCH_ZEN:
          case BLIS_ARCH_ZEN2:
          case BLIS_ARCH_ZEN3:
	      return bli_dgemm_tiny_6x8
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
	      break;
          default:
              return BLIS_FAILURE;
        }

    }

    if( FALSE == bli_thread_get_is_parallel() )
    {
        // Pick the kernel based on the architecture ID
        switch (arch_id)
        {
          case BLIS_ARCH_ZEN5:
          case BLIS_ARCH_ZEN4:
#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)
              if(((m == n) && (m < 400) && (k < 1000)) ||
              ( (m != n) && (( ((m + n -k) < 1500) &&
              ((m + k-n) < 1500) && ((n + k-m) < 1500) ) ||
              ((n <= 100) && (k <=100)))))
              {
                  return bli_dgemm_tiny_24x8
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
#endif
              break;

          case BLIS_ARCH_ZEN:
          case BLIS_ARCH_ZEN2:
          case BLIS_ARCH_ZEN3:
              if( ( (m <= 8)  || ( (m <= 1000) && (n <= 24) && (k >= 4) ) ) && (k <= 1500) )
              {
                  return bli_dgemm_tiny_6x8
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
              break;
          default:
              return BLIS_FAILURE;
        }
    }

    return BLIS_FAILURE;
}
