/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
/* ?gemm_compute.h */
/* BLAS interface to compute matrix-matrix product  */
/* Datatype : s & d (single and double precision only supported) */
/* BLAS Extensions */
/* output is the gemm result */

#include "blis.h"

void sgemm_compute_blis_impl
(
    const f77_char* transa,
    const f77_char* transb,
    const f77_int*  m,
    const f77_int*  n,
    const f77_int*  k,
    const float*    a, const f77_int* rs_a, const f77_int* cs_a,
    const float*    b, const f77_int* rs_b, const f77_int* cs_b,
    const float*    beta,
          float*    c, const f77_int* rs_c, const f77_int* cs_c
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    trans_t blis_transa;
    trans_t blis_transb;
    dim_t   m0, n0, k0;
    dim_t   m0_a, n0_a;
    dim_t   m0_b, n0_b;

    /* Initialize BLIS. */
    bli_init_auto();

    // @todo: Add AOCL DTL logs
    // AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    // AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(d), *transa, *transb, *m, *n, *k, 
                            //  (void*)alpha, *lda, *ldb, (void*)beta, *ldc);

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_compute)
    (
      MKSTR(s),
      MKSTR(gemm),
      transa,
      transb,
      m,
      n,
      k,
      ( ( *rs_a != 1 ) ? rs_a : cs_a ),
      ( ( *rs_b != 1 ) ? rs_b : cs_b ),
      rs_c, cs_c
    );

    /* Quick return. */
    if ( *m == 0 || *n == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_trans( *transb, &blis_transb );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1(*m, m0);
    bli_convert_blas_dim1(*n, n0);
    bli_convert_blas_dim1(*k, k0);

    const num_t dt = BLIS_FLOAT;

    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;
    obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co     = BLIS_OBJECT_INITIALIZER;

    bli_set_dims_with_trans( blis_transa, m0, k0, &m0_a, &n0_a );
    bli_set_dims_with_trans( blis_transb, k0, n0, &m0_b, &n0_b );

    bli_obj_init_finish_1x1( dt, (float*)beta,  &betao  );

    bli_obj_init_finish( dt, m0_a, n0_a, (float*)a, *rs_a, *cs_a, &ao );
    bli_obj_init_finish( dt, m0_b, n0_b, (float*)b, *rs_b, *cs_b, &bo );
    bli_obj_init_finish( dt, m0,   n0,   (float*)c, *rs_c, *cs_c, &co );

    bli_obj_set_conjtrans( blis_transa, &ao );
    bli_obj_set_conjtrans( blis_transb, &bo );

    PASTEMAC0( gemm_compute_init )
    (
        &ao,
        &bo,
        &betao,
        &co,
        NULL,
        NULL
    );

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return;
}

#ifdef BLIS_ENABLE_BLAS
void sgemm_compute_
(
    const f77_char* transa,
    const f77_char* transb,
    const f77_int*  m,
    const f77_int*  n,
    const f77_int*  k,
    const float*    a, const f77_int* lda,
    const float*    b, const f77_int* ldb,
    const float*    beta,
          float*    c, const f77_int* ldc
)
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    sgemm_compute_blis_impl( transa,
                             transb,
                             m,
                             n,
                             k,
                             a, &rs_a, lda,
                             b, &rs_b, ldb,
                             beta,
                             c, &rs_c, ldc );
}
#endif

void dgemm_compute_blis_impl
(
    const f77_char* transa,
    const f77_char* transb,
    const f77_int*  m,
    const f77_int*  n,
    const f77_int*  k,
    const double*   a, const f77_int* rs_a, const f77_int* cs_a,
    const double*   b, const f77_int* rs_b, const f77_int* cs_b,
    const double*   beta,
          double*   c, const f77_int* rs_c, const f77_int* cs_c
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    trans_t blis_transa;
    trans_t blis_transb;
    dim_t   m0, n0, k0;
    dim_t   m0_a, n0_a;
    dim_t   m0_b, n0_b;

    /* Initialize BLIS. */
    bli_init_auto();

    // @todo: Add AOCL DTL logs
    // AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    // AOCL_DTL_LOG_GEMM_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(d), *transa, *transb, *m, *n, *k, 
                            //  (void*)alpha, *lda, *ldb, (void*)beta, *ldc);

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_compute)
    (
      MKSTR(d),
      MKSTR(gemm),
      transa,
      transb,
      m,
      n,
      k,
      ( ( *rs_a != 1 ) ? rs_a : cs_a ),
      ( ( *rs_b != 1 ) ? rs_b : cs_b ),
      rs_c, cs_c
    );

   /* Quick return. */
    if ( *m == 0 || *n == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_trans( *transa, &blis_transa );
    bli_param_map_netlib_to_blis_trans( *transb, &blis_transb );

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1(*m, m0);
    bli_convert_blas_dim1(*n, n0);
    bli_convert_blas_dim1(*k, k0);

    const num_t dt = BLIS_DOUBLE;

    obj_t       ao     = BLIS_OBJECT_INITIALIZER;
    obj_t       bo     = BLIS_OBJECT_INITIALIZER;
    obj_t       betao  = BLIS_OBJECT_INITIALIZER_1X1;
    obj_t       co     = BLIS_OBJECT_INITIALIZER;

    bli_set_dims_with_trans( blis_transa, m0, k0, &m0_a, &n0_a );
    bli_set_dims_with_trans( blis_transb, k0, n0, &m0_b, &n0_b );

    bli_obj_init_finish_1x1( dt, (double*)beta,  &betao  );

    bli_obj_init_finish( dt, m0_a, n0_a, (double*)a, *rs_a, *cs_a, &ao );
    bli_obj_init_finish( dt, m0_b, n0_b, (double*)b, *rs_b, *cs_b, &bo );
    bli_obj_init_finish( dt, m0,   n0,   (double*)c, *rs_c, *cs_c, &co );

    bli_obj_set_conjtrans( blis_transa, &ao );
    bli_obj_set_conjtrans( blis_transb, &bo );

    PASTEMAC0( gemm_compute_init )
    (
        &ao,
        &bo,
        &betao,
        &co,
        NULL,
        NULL
    );

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return;
}

#ifdef BLIS_ENABLE_BLAS
void dgemm_compute_
(
    const f77_char* transa,
    const f77_char* transb,
    const f77_int*  m,
    const f77_int*  n,
    const f77_int*  k,
    const double*   a, const f77_int* lda,
    const double*   b, const f77_int* ldb,
    const double*   beta,
          double*   c, const f77_int* ldc
)
{
    f77_int rs_a = 1;
    f77_int rs_b = 1;
    f77_int rs_c = 1;
    dgemm_compute_blis_impl( transa,
                             transb,
                             m,
                             n,
                             k,
                             a, &rs_a, lda,
                             b, &rs_b, ldb,
                             beta,
                             c, &rs_c, ldc );
}
#endif
