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

// BLAS Extension APIs
/* ?gemm_pack_get_size.c */
/* This program is a C interface to query the size of storage */
/* required for a packed matrix structure to be used in subsequent calls */
/* Datatype : s & d (single and double precision only supported) */
/* BLAS Extensions */
/* returns number of bytes */

#include "blis.h"

f77_int dgemm_pack_get_size_blis_impl
     (
       const f77_char* identifier,
       const f77_int* pm,
       const f77_int* pn,
       const f77_int* pk
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    bli_init_auto(); // initialize blis
    cntx_t* cntx = bli_gks_query_cntx(); // Get processor specific context.

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_get_size)
    (
      MKSTR(d),
      MKSTR(gemm),
      identifier,
      pm,
      pn,
      pk
    );

    /* Quick return. */
    if ( *pm == 0 || *pn == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return 0;
    }

    num_t dt = BLIS_DOUBLE;  // Double precision
    f77_int tbytes   = 0;    // total number of bytes needed for packing.
    f77_int m = *pm;
    f77_int n = *pn;
    f77_int k = *pk;

    // Retrieve cache-blocking parameters used in GEMM

#if 0 // Not needed, MR and NR should do
    const dim_t MC  = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const dim_t KC  = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
    const dim_t NC  = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );

#endif

    const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );

    // Note: If one of the dimensions is zero - we return zero bytes

    // if we allocate memory based on MC, KC and NC. We might be wasting the memory
    // for matrix sizes smaller than these values
    // When packing A - MC x KC is size of a row-panel but when k < KC then
    // we take only MC x k. Basically the row-panel size is MC x min(k, KC).
    // But if m < MC. Then we make "m" aligned to MR. m_p_pad = m aligned to MR.
    // Minimum unit of work the Kernel operates is by computing MR x NR  block of C.
    // Kernel: It multiplies MR x min(k, KC) of A column-micro panel with min(k, KC) x NR
    // of B row-micro panel.

    // Therefore the packing sizes will be :
    // For A pack - m_p_pad x k. where m_p_pad = m multiple of MR.
    // For B pack - k x n_p_pad. where n_p_pad = n multiple of NR.

    if ( (*identifier == 'a') || (*identifier == 'A') )
    {
        // Size of single packed A buffer is MC x KC elements - row-panels of A
        // Number of elements in row-panel of A = MC x KC
        // size of micro-panels is MR x KC
        dim_t m_p_pad = ( (m + MR - 1)/MR ) * MR;
        dim_t ps_n = m_p_pad * k; //  size of all packed buffer (multiples of MR x k)

        // if A is transposed - then A' dimensions will be k x m
        // here k should be multiple of MR
        dim_t mt_p_pad = ((k + MR -1)/MR ) * MR;

        dim_t ps_t = mt_p_pad * m;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof( double );
    }
    else if ( (*identifier == 'b') || (*identifier == 'B') )
    {
        // Size of Single Packed B buffer is KC x NC elements. - Column panels of B
        // Number of elements in column-panel of B = KC x NC

        // size of micro-panels is KC x NR
        dim_t n_p_pad = ( (n + NR - 1)/NR ) * NR;
        dim_t ps_n    = k * n_p_pad; // size of packed buffer of B (multiples of k x NR)

        // if B is transposed then B' - dimension is n x k
        // here k should be multiple of NR
        dim_t nt_p_pad = ( (k + NR -1)/NR ) * NR;
        dim_t ps_t = n * nt_p_pad;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof( double );
    }

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return tbytes;
}

#ifdef BLIS_ENABLE_BLAS
f77_int dgemm_pack_get_size_
     (
       const f77_char* identifier,
       const f77_int* pm,
       const f77_int* pn,
       const f77_int* pk
     )
{
    return dgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}
#endif

f77_int sgemm_pack_get_size_blis_impl
     (
       const f77_char* identifier,
       const f77_int* pm,
       const f77_int* pn,
       const f77_int* pk
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    bli_init_auto(); // initialize blis
    cntx_t* cntx = bli_gks_query_cntx(); // Get processor specific context.

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_get_size)
    (
      MKSTR(s),
      MKSTR(gemm),
      identifier,
      pm,
      pn,
      pk
    );

    /* Quick return. */
    if ( *pm == 0 || *pn == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return 0;
    }

    num_t dt = BLIS_FLOAT;  // Single precision
    f77_int tbytes   = 0;    // total number of bytes needed for packing.
    f77_int m = *pm;
    f77_int n = *pn;
    f77_int k = *pk;

    // Retrieve cache-blocking parameters used in GEMM

#if 0 // Not needed, MR and NR should do
    const dim_t MC  = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const dim_t KC  = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
    const dim_t NC  = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );

#endif

    const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );

    // Note: If one of the dimensions is zero - we return zero bytes

    // if we allocate memory based on MC, KC and NC. We might be wasting the memory
    // for matrix sizes smaller than these values
    // When packing A - MC x KC is size of a row-panel but when k < KC then
    // we take only MC x k. Basically the row-panel size is MC x min(k, KC).
    // But if m < MC. Then we make "m" aligned to MR. m_p_pad = m aligned to MR.
    // Minimum unit of work the Kernel operates is by computing MR x NR  block of C.
    // Kernel: It multiplies MR x min(k, KC) of A column-micro panel with min(k, KC) x NR
    // of B row-micro panel.

    // Therefore the packing sizes will be :
    // For A pack - m_p_pad x k. where m_p_pad = m multiple of MR.
    // For B pack - k x n_p_pad. where n_p_pad = n multiple of NR.

    if ( (*identifier == 'a') || (*identifier == 'A') )
    {
        // Size of single packed A buffer is MC x KC elements - row-panels of A
        // Number of elements in row-panel of A = MC x KC
        // size of micro-panels is MR x KC
        dim_t m_p_pad = ( (m + MR - 1)/MR ) * MR;
        dim_t ps_n = m_p_pad * k; //  size of all packed buffer (multiples of MR x k)

        // if A is transposed - then A' dimensions will be k x m
        // here k should be multiple of MR
        dim_t mt_p_pad = ((k + MR -1)/MR ) * MR;

        dim_t ps_t = mt_p_pad * m;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof( float );
    }
    else if ( (*identifier == 'b') || (*identifier == 'B'))
    {
        // Size of Single Packed B buffer is KC x NC elements. - Column panels of B
        // Number of elements in column-panel of B = KC x NC

        // size of micro-panels is KC x NR
        dim_t n_p_pad = ( (n + NR - 1)/NR ) * NR;
        dim_t ps_n    = k * n_p_pad; // size of packed buffer of B (multiples of k x NR)

        // if B is transposed then B' - dimension is n x k
        // here k should be multiple of NR
        dim_t nt_p_pad = ( (k + NR -1)/NR ) * NR;
        dim_t ps_t = n * nt_p_pad;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof( float );
    }

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return tbytes;
}

#ifdef BLIS_ENABLE_BLAS
f77_int sgemm_pack_get_size_
     (
       const f77_char* identifier,
       const f77_int* pm,
       const f77_int* pn,
       const f77_int* pk
     )
{
    return sgemm_pack_get_size_blis_impl( identifier, pm, pn, pk );
}
#endif
