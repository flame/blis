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
/* ?gemm_pack.h */
/* BLAS interface to perform scaling and packing of the  */
/* matrix  to a packed matrix structure to be used in subsequent calls */
/* Datatype : s & d (single and double precision only supported) */
/* BLAS Extensions */

#include "blis.h"

void sgemm_pack_blis_impl
     (
       const f77_char* identifier,
       const f77_char* trans,
       const f77_int*  mm,
       const f77_int*  nn,
       const f77_int*  kk,
       const float*    alpha,
       const float*    src, const f77_int*  pld,
             float*    dest
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    dim_t m;
    dim_t n;
    dim_t k;

    bli_init_auto(); // initialize blis

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_pack)
    (
      MKSTR(s),
      MKSTR(gemm),
      identifier,
      trans,
      mm,
      nn,
      kk,
      pld
    );

    /* Quick return. */
    if ( *mm == 0 || *nn == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

    dim_t m0 = 0;
    dim_t n0 = 0;

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *mm, m );
    bli_convert_blas_dim1( *nn, n );
    bli_convert_blas_dim1( *kk, k );

    inc_t cs = *pld;
    inc_t rs = 1;

    trans_t blis_trans;

    num_t dt = BLIS_FLOAT;

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_trans( *trans, &blis_trans );

    obj_t src_obj = BLIS_OBJECT_INITIALIZER;
    obj_t dest_obj = BLIS_OBJECT_INITIALIZER;
    obj_t alpha_obj = BLIS_OBJECT_INITIALIZER;


    if (*identifier == 'a' || *identifier == 'A')
    {
        bli_set_dims_with_trans( blis_trans, m, k, &m0, &n0 );
    }
    else if (*identifier == 'b' || *identifier == 'B')
    {
        bli_set_dims_with_trans( blis_trans, k, n, &m0, &n0 );
    }

    bli_obj_init_finish_1x1( dt, (float*)alpha,  &alpha_obj );

    bli_obj_init_finish( dt, m0, n0, (float*)src, rs, cs, &src_obj );
    bli_obj_init_finish( dt, m0, n0, (float*)dest, rs, cs, &dest_obj );

    bli_obj_set_conjtrans( blis_trans, &src_obj );

    bli_pack_full_init(identifier, &alpha_obj, &src_obj, &dest_obj, NULL, NULL);

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return;
}

void sgemm_pack_
     (
       const f77_char* identifier,
       const f77_char* trans,
       const f77_int*  mm,
       const f77_int*  nn,
       const f77_int*  kk,
       const float*    alpha,
       const float*    src, const f77_int*  pld,
             float*    dest
     )
{
    sgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}

void dgemm_pack_blis_impl
     (
       const f77_char* identifier,
       const f77_char* trans,
       const f77_int*  mm,
       const f77_int*  nn,
       const f77_int*  kk,
       const double*   alpha,
       const double*   src, const f77_int*  pld,
             double*   dest
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);

    dim_t m;
    dim_t n;
    dim_t k;

    bli_init_auto(); // initialize blis

    /* Perform BLAS parameter checking. */
    PASTEBLACHK(gemm_pack)
    (
      MKSTR(d),
      MKSTR(gemm),
      identifier,
      trans,
      mm,
      nn,
      kk,
      pld
    );

    /* Quick return. */
    if ( *mm == 0 || *nn == 0 )
    {
      /* Finalize BLIS. */
      bli_finalize_auto();
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

    dim_t m0 = 0;
    dim_t n0 = 0;

    /* Typecast BLAS integers to BLIS integers. */
    bli_convert_blas_dim1( *mm, m );
    bli_convert_blas_dim1( *nn, n );
    bli_convert_blas_dim1( *kk, k );

    inc_t cs = *pld;
    inc_t rs = 1;

    trans_t blis_trans;

    num_t dt = BLIS_DOUBLE;

    /* Map BLAS chars to their corresponding BLIS enumerated type value. */
    bli_param_map_netlib_to_blis_trans( *trans, &blis_trans );

    obj_t src_obj = BLIS_OBJECT_INITIALIZER;
    obj_t dest_obj = BLIS_OBJECT_INITIALIZER;
    obj_t alpha_obj = BLIS_OBJECT_INITIALIZER;

    if (*identifier == 'a' || *identifier == 'A')
    {
        bli_set_dims_with_trans( blis_trans, m, k, &m0, &n0 );
    }
    else if (*identifier == 'b' || *identifier == 'B')
    {
        bli_set_dims_with_trans( blis_trans, k, n, &m0, &n0 );
    }

    bli_obj_init_finish_1x1( dt, (double*)alpha,  &alpha_obj );

    bli_obj_init_finish( dt, m0, n0, (double*)src, rs, cs, &src_obj );
    bli_obj_init_finish( dt, m0, n0, (double*)dest, rs, cs, &dest_obj );

    bli_obj_set_conjtrans( blis_trans, &src_obj );

    bli_pack_full_init(identifier, &alpha_obj, &src_obj, &dest_obj, NULL, NULL);

    /* Finalize BLIS. */
    bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return;
}

void dgemm_pack_
     (
       const f77_char* identifier,
       const f77_char* trans,
       const f77_int*  mm,
       const f77_int*  nn,
       const f77_int*  kk,
       const double*   alpha,
       const double*   src, const f77_int*  pld,
             double*   dest
     )
{
    dgemm_pack_blis_impl( identifier, trans, mm, nn, kk, alpha, src, pld, dest );
}
