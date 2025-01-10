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

/**
 * Currently buffer size is defined in bytes as per
 * the tiny path threshold.
 *
 * TODO: If the threshold for tiny gemm are changed, the buffer size needs
 * to be updated respectively. Such interface change will be incorporated with light
 * weight memory pool implementation.
 */
#define PACK_BUFFER_SIZE_B 1499 * 1499 * 8

/**
 * TODO: Blocking related changes yet to be added.
 * Which involves, separate call to kernel scaling of C matrix
 * by actual Beta once for first iteration of K.
 *
 * Following K iteration uses Beta=1.0 where we need to call
 * another set of kernels which instead of using FMAs while
 * scaling by Beta and adding Alpha*A*B to C matrix, it simply can
 * use vmuladdpd instruction.
 */

/**
 * @brief
 *
 * Here based N dimension, it is decided whether main kernel needs to be invoked
 * or simply jump straight to edge kernel directly.
 * All the main kernel + remaining 7 edge kernels are registered in kernel table,
 * which is nothing but table of function pointer and each index represents N.
 *
 * For N >= 8, main kernel is invoked. Remainder cases of N are handled from within.
 * For N < 8, direclt N specific edge kernel is invoked for gemm computation.
 * @note
 * N = 0 case never occurs.
 */
#define CALL_KERNEL\
        if(N >= 8)\
        {\
            avx512kern_fp[8](   conja,\
                                conjb,\
                                M,\
                                N,\
                                K,\
                                (double *)alpha,\
                                (a_local + (0 * rs_a) + (0 * cs_a)), /*A matrix offset*/\
                                rs_a,\
                                cs_a,\
                                (b_local + (0 * cs_b) + (0 * rs_b)), /*B matrix offset*/\
                                rs_b,\
                                cs_b,\
                                (double *)beta,\
                                (c_local + 0 * cs_c + 0 * rs_c),     /*C matrix offset*/\
                                rs_c,\
                                cs_c,\
                                &aux,\
                                NULL\
                            );\
        }\
        else\
        {\
            avx512kern_fp[N](   conja,\
                                conjb,\
                                M,\
                                N,\
                                K,\
                                (double *)alpha,\
                                (a_local + (0 * rs_a) + (0 * cs_a)), /*A matrix offset*/\
                                rs_a,\
                                cs_a,\
                                (b_local + (0 * cs_b) + (0 * rs_b)), /*B matrix offset*/\
                                rs_b,\
                                cs_b,\
                                (double *)beta,\
                                (c_local + 0 * cs_c + 0 * rs_c),     /*C matrix offset*/\
                                rs_c,\
                                cs_c,\
                                &aux,\
                                NULL\
                            );\
        }
/**
 * @brief bli_dgemmsup_placeholder
 * 
 * This is just a dummy function, which does nothing.
 * Instead of setting 0th index avx512kern_fp table to NULL,
 * this dummy function is assigned.
 * Since we are directly calling kernels via function pointer
 * without null checks, it is better to assign a dummy function
 * rather than NULL, which may lead to crash.
 */
static void bli_dgemmsup_placeholder
     (
        conj_t    conja,
        conj_t    conjb,
        dim_t     m0,
        dim_t     n0,
        dim_t     k0,
        double*    restrict alpha,
        double*    restrict a, inc_t rs_a, inc_t cs_a,
        double*    restrict b, inc_t rs_b, inc_t cs_b,
        double*    restrict beta,
        double*    restrict c, inc_t rs_c, inc_t cs_c,
        auxinfo_t* restrict data,
        cntx_t*    restrict cntx
     )
{
    return;
}

static dgemmsup_ker_ft avx512kern_fp[] =
{
    bli_dgemmsup_placeholder,
    bli_dgemmsup_rv_zen4_asm_24x1m_new,
    bli_dgemmsup_rv_zen4_asm_24x2m_new,
    bli_dgemmsup_rv_zen4_asm_24x3m_new,
    bli_dgemmsup_rv_zen4_asm_24x4m_new,
    bli_dgemmsup_rv_zen4_asm_24x5m_new,
    bli_dgemmsup_rv_zen4_asm_24x6m_new,
    bli_dgemmsup_rv_zen4_asm_24x7m_new,
    bli_dgemmsup_rv_zen4_asm_24x8m_new
};


err_t bli_dgemm_tiny_24x8
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
         * swap A and B matrix.
         */
        a_local = (double *)b;
        b_local = (double *)a;

        /**
         * The row stride (rs_a) and column stride (cs_b) of matrix A and B are swapped using XOR swap.
         * The column stride (cs_a) and row stride (rs_b) of matrix A and B are swapped using XOR swap.
         */
        rs_a ^= cs_b;
        cs_b ^= rs_a;
        rs_a ^= cs_b;

        cs_a ^= rs_b;
        rs_b ^= cs_a;
        cs_a ^= rs_b;

        /**
         * The row stride (rs_c) and column stride (cs_c) of matrix C are swapped using XOR swap.
         */
        rs_c ^= cs_c;
        cs_c ^= rs_c;
        rs_c ^= cs_c;

        /**
         * The dimensions M and N are swapped.
         */
        M = n;
        N = m;
    }

    auxinfo_t aux;
    inc_t ps_a_use = (24 * rs_a);
    bli_auxinfo_set_ps_a( ps_a_use, &aux );

    if(stor_id == BLIS_CRC || stor_id == BLIS_RRC)
    {
        double *a_pkr = NULL;

	/**
	 * @brief
	 * When BLIS_PACK_BUFFER macro is set to 1, we inquire and use buffer
	 * allocated from blis memory pool.
	 * Here the size is given hard-coded, reason behind it is as explain at the defination
	 * of local static packA_buffer.
	 * Once the buffer is acquired and after sanity check, it packs A matrix in column stored fashion
	 * and pass it cv kernel for computation.
	 * Here the macro CALL_KERNEL is final call to the kernel.
	 * Reason for having separate call to CALL_KERNEL inside the if..else.. condition is, when BLIS_PACK_BUFFER
	 * is enabled we acquire packing buffer from blis memory pool, which is also an already allocated static aligned
	 * memory buffer under the hood. So after using it needs to be returned to pool so if do not have separate conditions
	 * It will end up checking for allocated buffer even for the cases, where we do not even allocate a buffer to pack
	 * A matrix. Such as any storage scheme other than CRC and RRC.
	 */
        mem_t local_mem_buf_A_s;
        rntm_t rntm;
        bli_pba_rntm_set_pba( &rntm );

        // Get the buffer from the pool.
        bli_pba_acquire_m(&rntm,
                                PACK_BUFFER_SIZE_B,
                                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                                &local_mem_buf_A_s);

        double *packA_buffer = bli_mem_buffer(&local_mem_buf_A_s);

        if(packA_buffer == NULL)
        {
            return BLIS_FAILURE;
        }

	a_pkr = packA_buffer;

        dim_t m_iter = M /24;
        dim_t m_left = M % 24;

        double *a_ptr = a_local;
        double local_one = 1.0;

        for(int i = 0; i < m_iter; i++)
        {
            bli_dpackm_zen4_asm_24xk(BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS, 24, k, k, &local_one, a_ptr, rs_a, 1, a_pkr, 24, NULL);
            a_ptr += 24 * rs_a;
            a_pkr += 24 * k;
        }

        if(m_left)
        {
            bli_dpackm_zen4_asm_24xk(BLIS_NO_CONJUGATE, BLIS_PACKED_ROWS, m_left, k, k, &local_one, a_ptr, rs_a, 1, a_pkr, 24, NULL);
        }

        a_local = packA_buffer;
        rs_a = 1;
        cs_a = 24;
        ps_a_use = (24 * k);
        bli_auxinfo_set_ps_a( ps_a_use, &aux );

        CALL_KERNEL

	//Return the allocated memory back to small block allocator
        bli_pba_release(&rntm, &local_mem_buf_A_s);
    }
    else
    {
        CALL_KERNEL
    }
    return BLIS_SUCCESS;
}

