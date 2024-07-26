/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include "xmmintrin.h"
#include "blis.h"

static dgemmsup_ker_ft kern_fp[] =
{
    bli_dgemmsup_rv_haswell_asm_6x8m,
    bli_dgemmsup_rd_haswell_asm_6x8m,
    bli_dgemmsup_rv_haswell_asm_6x8m,
    bli_dgemmsup_rv_haswell_asm_6x8n,
    bli_dgemmsup_rv_haswell_asm_6x8m,
    bli_dgemmsup_rd_haswell_asm_6x8n,
    bli_dgemmsup_rv_haswell_asm_6x8n,
    bli_dgemmsup_rv_haswell_asm_6x8n
};

#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)
static err_t bli_dgemm_tiny_24x8_kernel
     (
        conj_t              conja,
        conj_t              conjb,
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
    double *a_local = (double *)a;
    double *b_local = (double *)b;
    double *c_local = (double *)c;
    guint_t cs_a = cs_a0;
    guint_t rs_a = rs_a0;
    guint_t cs_b = cs_b0;
    guint_t rs_b = rs_b0;
    guint_t cs_c = cs_c0;
    guint_t rs_c = rs_c0;
    inc_t rs_a_local = rs_a0;
    inc_t cs_a_local = cs_a0;
    inc_t rs_b_local = rs_b0;
    inc_t cs_b_local = cs_b0;
    inc_t rs_c_local = rs_c0;
    inc_t cs_c_local = cs_c0;

    gint_t M = m;
    gint_t N = n;
    gint_t K = k;

    inc_t storage = 0;
    if(transb == BLIS_NO_TRANSPOSE || transb == BLIS_CONJ_NO_TRANSPOSE)
    {
        storage = 1 * (rs_b == 1);     //1st bit
    }
    else if(transb == BLIS_TRANSPOSE || transb == BLIS_CONJ_TRANSPOSE)
    {
        storage = 1 * (cs_b == 1);     //1st bit
        rs_b = cs_b0;
        cs_b = rs_b0;
    }

    if(transa == BLIS_NO_TRANSPOSE || transa == BLIS_CONJ_NO_TRANSPOSE)
    {
        storage |= ((1 * (rs_a == 1)) << 1); //2nd bit
    }
    else if(transa == BLIS_TRANSPOSE || transa == BLIS_CONJ_TRANSPOSE)
    {
        storage |= ((1 * (cs_a == 1)) << 1); //2nd bit
        rs_a = cs_a0;
        cs_a = rs_a0;
    }

    storage |= ((1 * (rs_c == 1)) << 2); //3rd bit

    stor3_t stor_id = (stor3_t) storage;

    //Early return, since we do not support dot product gemm kernels.
    if(stor_id == BLIS_CRC || stor_id == BLIS_RRC)
    {
        return BLIS_FAILURE;
    }

    const bool is_rrr_rrc_rcr_crr = (
                                    stor_id == BLIS_RRR ||
                                    stor_id == BLIS_RRC ||
                                    stor_id == BLIS_RCR ||
                                    stor_id == BLIS_CRR
                                    );

    const bool is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;
    const bool row_pref = false;
    const bool col_pref = !row_pref;

    const bool is_primary = ( row_pref && is_rrr_rrc_rcr_crr ) ||
                            ( col_pref && is_rcc_crc_ccr_ccc );

   /**
    * Based on matrix storage scheme and kernel preference,
    * decision is made here that whether it is primary storage
    * scheme or not.
    */
   if ( !is_primary )
    {
	/**
	 * For non-primary storage scheme, we configure parameters,
	 * for kernel re-use.
	 */
        a_local = (double *)b;
        b_local = (double *)a;
        rs_a_local = cs_b;
        cs_a_local = rs_b;
        rs_b_local = cs_a;
        cs_b_local = rs_a;
        rs_c_local = cs_c0;
        cs_c_local = rs_c0;
        M = n;
        N = m;

        rs_a = rs_a_local;
        cs_a = cs_a_local;
        cs_c = cs_c_local;
        rs_b = rs_b_local;
        cs_b = cs_b_local;
        rs_c = rs_c_local;
    }

    double *A = a_local;
    double *B = b_local;
    double *C = c_local;
    double *alpha_cast;
    double beta_cast = *beta;
    double one_local = 1.0;
    alpha_cast = (double *)alpha;
    /**
     * Set blocking and micro tile parameters before computing
     */
    const dim_t MC = 144;
    const dim_t KC = 480;
    const dim_t MR_ = 24;
    const dim_t NR_ = 8;
    /**
     * MC must be in multiple of MR_.
     * if not return early.
    */
    if( MC % MR_ != 0 )
    {
        return BLIS_FAILURE;
    }
    dim_t n_rem = N % NR_;
    dim_t m_part_rem = M % MC;
    dim_t k_rem = K % KC;
    dim_t n_cur  = 0;
    dim_t m_cur  = 0;
    dim_t k_cur  = 0;
    dim_t k_iter = 0;
    auxinfo_t aux;
    inc_t ps_a_use = (MR_ * rs_a);
    bli_auxinfo_set_ps_a( ps_a_use, &aux );

    /**
     * JC Loop is eliminated as it iterates only once, So computation
     * can start from K loop.
     * Here K loop is divided into two parts to avoid repetitive check for Beta.
     * For first iteration, it will use Beta to scale C matrix.
     * Subsequent iterations will scale C matrix by 1.
     */
    k_iter = 0; //1st k loop, scale C matrix by beta
    k_cur = (KC <= K ? KC : k_rem);
    for ( dim_t m_iter = 0; m_iter < M;  m_iter += MC)
    {
        m_cur = (MC <= (M - m_iter) ? MC : m_part_rem);
        for ( dim_t jr_iter = 0; jr_iter < N; jr_iter += NR_ )
        {
            n_cur = (NR_ <= (N - jr_iter) ? NR_ : n_rem);
            bli_dgemmsup_rv_zen4_asm_24x8m(conja,
                                           conjb,
                                           m_cur,
                                           n_cur,
                                           k_cur,
                                           alpha_cast,
                                           (A + (m_iter * rs_a) + (k_iter * cs_a)), /*A matrix offset*/
                                           rs_a,
                                           cs_a,
                                           (B + (jr_iter * cs_b) + (k_iter * rs_b)), /*B matrix offset*/
                                           rs_b,
                                           cs_b,
                                           &beta_cast,
                                           (C + jr_iter * cs_c + m_iter * rs_c), /*C matrix offset*/
                                           rs_c,
                                           cs_c,
                                           &aux,
                                           NULL);
        }
    }
    // k_iter = KC loop where C matrix is scaled by one. Beta is one.
    for (k_iter = KC ; k_iter < K; k_iter += KC )
    {
        k_cur = (KC <= (K - k_iter) ? KC : k_rem);
        for ( dim_t m_iter = 0; m_iter < M;  m_iter += MC)
        {
            m_cur = (MC <= (M - m_iter) ? MC : m_part_rem);
            for ( dim_t jr_iter = 0; jr_iter < N; jr_iter += NR_ )
            {
                n_cur = (NR_ <= (N - jr_iter) ? NR_ : n_rem);
                bli_dgemmsup_rv_zen4_asm_24x8m(conja,
                                               conjb,
                                               m_cur,
                                               n_cur,
                                               k_cur,
                                               alpha_cast,
                                               (A + (m_iter * rs_a) + (k_iter * cs_a)), /*A matrix offset*/
                                               rs_a,
                                               cs_a,
                                               (B + (jr_iter * cs_b) + (k_iter * rs_b)), /*B matrix offset*/
                                               rs_b,
                                               cs_b,
                                               &one_local,
                                               (C + jr_iter * cs_c + m_iter * rs_c), /*C matrix offset*/
                                               rs_c,
                                               cs_c,
                                               &aux,
                                               NULL);
            }
        }
    }

    return BLIS_SUCCESS;
}
#endif

static err_t bli_dgemm_tiny_6x8_kernel
     (
        conj_t              conja,
        conj_t              conjb,
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
    double *a_local = (double *)a;
    double *b_local = (double *)b;
    double *c_local = (double *)c;
    guint_t cs_a = cs_a0;
    guint_t rs_a = rs_a0;
    guint_t cs_b = cs_b0;
    guint_t rs_b = rs_b0;
    guint_t cs_c = cs_c0;
    guint_t rs_c = rs_c0;
    inc_t rs_a_local = rs_a0;
    inc_t cs_a_local = cs_a0;
    inc_t rs_b_local = rs_b0;
    inc_t cs_b_local = cs_b0;
    inc_t rs_c_local = rs_c0;
    inc_t cs_c_local = cs_c0;

    gint_t M = m;
    gint_t N = n;
    gint_t K = k;

    inc_t storage = 0;
    if(transb == BLIS_NO_TRANSPOSE || transb == BLIS_CONJ_NO_TRANSPOSE)
    {
        storage = 1 * (rs_b == 1);     //1st bit
    }
    else if(transb == BLIS_TRANSPOSE || transb == BLIS_CONJ_TRANSPOSE)
    {
        storage = 1 * (cs_b == 1);     //1st bit
        rs_b = cs_b0;
        cs_b = rs_b0;
    }

    if(transa == BLIS_NO_TRANSPOSE || transa == BLIS_CONJ_NO_TRANSPOSE)
    {
        storage |= ((1 * (rs_a == 1)) << 1); //2nd bit
    }
    else if(transa == BLIS_TRANSPOSE || transa == BLIS_CONJ_TRANSPOSE)
    {
        storage |= ((1 * (cs_a == 1)) << 1); //2nd bit
        rs_a = cs_a0;
        cs_a = rs_a0;
    }

    storage |= ((1 * (rs_c == 1)) << 2); //3rd bit

   /**
    * typecast storage into stor_idd,
    * stores default storage scheme before we optimze
    * for respective gemm kernel.    */
    stor3_t stor_idd = (stor3_t) storage;
    stor3_t stor_id = 0;

    stor_id = stor_idd;

    const bool is_rrr_rrc_rcr_crr = (
                                    stor_idd == BLIS_RRR ||
                                    stor_idd == BLIS_RRC ||
                                    stor_idd == BLIS_RCR ||
                                    stor_idd == BLIS_CRR
                                    );

    const bool is_rcc_crc_ccr_ccc = !is_rrr_rrc_rcr_crr;
    const bool row_pref = true;
    const bool col_pref = !row_pref;

   /**
    * Based on matrix storage scheme and kernel preference,
    * decision is made here that whether it is primary storage
    * scheme or not.
    */
    const bool is_primary = ( row_pref && is_rrr_rrc_rcr_crr ) ||
                            ( col_pref && is_rcc_crc_ccr_ccc );

   /**
    * For non-primary storage scheme, we configure parameters,
    * for kernel re-use.
    */
   if ( !is_primary )
    {
        a_local = (double *)b;
        b_local = (double *)a;
        rs_a_local = cs_b;
        cs_a_local = rs_b;
        rs_b_local = cs_a;
        cs_b_local = rs_a;
        rs_c_local = cs_c0;
        cs_c_local = rs_c0;
        M = n;
        N = m;

        stor_id = bli_stor3_trans(stor_idd);

        rs_a = rs_a_local;
        cs_a = cs_a_local;
        cs_c = cs_c_local;
        rs_b = rs_b_local;
        cs_b = cs_b_local;
        rs_c = rs_c_local;
    }

    double *A = a_local;
    double *B = b_local;
    double *C = c_local;
    double *alpha_cast;
    double beta_cast = *beta;
    double one_local = 1.0;

    alpha_cast = (double *)alpha;
    /**
     * Set blocking and micro tile parameters before computing
     */
    const dim_t MC = 72;
    const dim_t KC = 256;
    const dim_t MR_ = 6;
    const dim_t NR_ = 8;


    /**
     * MC must be in multiple of MR_.
     * if not return early.
    */
    if( MC % MR_ != 0 )
    {
        return BLIS_FAILURE;
    }
    dim_t n_rem = N % NR_;
    dim_t m_part_rem = M % MC;
    dim_t k_rem = K % KC;
    dim_t n_cur  = 0;
    dim_t m_cur  = 0;
    dim_t k_cur  = 0;
    dim_t k_iter = 0;

    auxinfo_t aux;
    inc_t ps_a_use = (MR_ * rs_a);
    bli_auxinfo_set_ps_a( ps_a_use, &aux );
    dgemmsup_ker_ft kern_ptr = kern_fp[stor_id];

    /**
     * JC Loop is eliminated as it iterates only once, So computation
     * can start from K loop.
     * Here K loop is divided into parts to avoid repetitive check for Beta.
     * For first iteration, it will use Beta to scale C matrix.
     * Subsequent iterations will scale C matrix by 1.
     */
    k_iter = 0; //1st k loop, scale C matrix by beta
    k_cur = (KC <= K ? KC : k_rem);
    for ( dim_t m_iter = 0; m_iter < M;  m_iter += MC)
    {
        m_cur = (MC <= (M - m_iter) ? MC : m_part_rem);
        for ( dim_t jr_iter = 0; jr_iter < N; jr_iter += NR_ )
        {
            n_cur = (NR_ <= (N - jr_iter) ? NR_ : n_rem);
            kern_ptr(conja,
                     conjb,
                     m_cur,
                     n_cur,
                     k_cur,
                     alpha_cast,
                     (A + (m_iter * rs_a) + (k_iter * cs_a)), /*A matrix offset*/
                     rs_a,
                     cs_a,
                     (B + (jr_iter * cs_b) + (k_iter * rs_b)), /*B matrix offset*/
                     rs_b,
                     cs_b,
                     &beta_cast,
                     (C + (jr_iter * cs_c) + (m_iter * rs_c)), /*C matrix offset*/
                     rs_c,
                     cs_c,
                     &aux,
                     NULL);
        }
    }
    // k_iter = KC loop where C matrix is scaled by one. Beta is one.
    for (k_iter = KC; k_iter < K; k_iter += KC )
    {
        k_cur = (KC <= (K - k_iter) ? KC : k_rem);
        for ( dim_t m_iter = 0; m_iter < M;  m_iter += MC)
        {
            m_cur = (MC <= (M - m_iter) ? MC : m_part_rem);
            for ( dim_t jr_iter = 0; jr_iter < N; jr_iter += NR_ )
            {
                n_cur = (NR_ <= (N - jr_iter) ? NR_ : n_rem);
                kern_ptr(conja,
                         conjb,
                         m_cur,
                         n_cur,
                         k_cur,
                         alpha_cast,
                         (A + (m_iter * rs_a) + (k_iter * cs_a)), /*A matrix offset*/
                         rs_a,
                         cs_a,
                         (B + (jr_iter * cs_b) + (k_iter * rs_b)), /*B matrix offset*/
                         rs_b,
                         cs_b,
                         &one_local,
                         (C + (jr_iter * cs_c) + (m_iter * rs_c)), /*C matrix offset*/
                         rs_c,
                         cs_c,
                         &aux,
                         NULL);
            }
        }
    }

    return BLIS_SUCCESS;
}

static arch_t get_arch_id(void)
{
    static arch_t arch_id = BLIS_NUM_ARCHS + 1;
    if(arch_id == BLIS_NUM_ARCHS + 1)
    {
        arch_id = bli_cpuid_query_id();
    }

    return arch_id;
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
    arch_t id = bli_arch_query_id();

        // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
            if(m<24 && ((n<=24 && k<=20) ||
            (n<=50 && ((m<=4 && k<=50) || (m!=8 && m!=9 && m!=16 && k<=10)))))
            {
                return bli_dgemm_tiny_6x8_kernel
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
        case BLIS_ARCH_ZEN4:
        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:
            if(m <= 24 && n <= 24 && k <= 20)
            {
                return bli_dgemm_tiny_6x8_kernel
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

    if(FALSE == bli_thread_get_is_parallel())
    {
        // Pick the kernel based on the architecture ID
        switch (id)
        {
          case BLIS_ARCH_ZEN5:
          case BLIS_ARCH_ZEN4:
#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)
              if(((m == n) && (m < 400) && (k < 1000)) ||
              ( (m != n) && (( ((m + n -k) < 1500) &&
              ((m + k-n) < 1500) && ((n + k-m) < 1500) ) ||
              ((n <= 100) && (k <=100)))))
              {
                  return bli_dgemm_tiny_24x8_kernel
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
                  return bli_dgemm_tiny_6x8_kernel
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
