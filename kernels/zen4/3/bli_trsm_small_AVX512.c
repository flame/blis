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
#include "blis.h"

#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
#include "immintrin.h"
#define BLIS_ENABLE_PREFETCH_IN_TRSM_SMALL

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
  #define DIAG_ELE_INV_OPS(a, b) (a / b)
  #define DIAG_ELE_EVAL_OPS(a, b) (a * b)
#endif

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
  #define DIAG_ELE_INV_OPS(a, b) (a * b)
  #define DIAG_ELE_EVAL_OPS(a, b) (a / b)
#endif

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
  #define DTRSM_SMALL_DIV_OR_SCALE _mm256_div_pd
  #define DTRSM_SMALL_DIV_OR_SCALE_AVX512 _mm512_div_pd
#endif

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
  #define DTRSM_SMALL_DIV_OR_SCALE _mm256_mul_pd
  #define DTRSM_SMALL_DIV_OR_SCALE_AVX512 _mm512_mul_pd
#endif

#define BLIS_SET_YMM_REG_ZEROS_AVX512 \
  ymm0  = _mm256_setzero_pd(); \
  ymm1  = _mm256_setzero_pd(); \
  ymm2  = _mm256_setzero_pd(); \
  ymm3  = _mm256_setzero_pd(); \
  ymm4  = _mm256_setzero_pd(); \
  ymm5  = _mm256_setzero_pd(); \
  ymm6  = _mm256_setzero_pd(); \
  ymm7  = _mm256_setzero_pd(); \
  ymm8  = _mm256_setzero_pd(); \
  ymm9  = _mm256_setzero_pd(); \
  ymm10 = _mm256_setzero_pd(); \
  ymm11 = _mm256_setzero_pd(); \
  ymm12 = _mm256_setzero_pd(); \
  ymm13 = _mm256_setzero_pd(); \
  ymm14 = _mm256_setzero_pd(); \
  ymm15 = _mm256_setzero_pd(); \
  ymm16 = _mm256_setzero_pd(); \
  ymm17 = _mm256_setzero_pd(); \
  ymm18 = _mm256_setzero_pd(); \
  ymm19 = _mm256_setzero_pd(); \
  ymm20 = _mm256_setzero_pd(); \
  ymm21 = _mm256_setzero_pd(); \
  ymm22 = _mm256_setzero_pd(); \
  ymm23 = _mm256_setzero_pd(); \
  ymm24 = _mm256_setzero_pd(); \
  ymm25 = _mm256_setzero_pd(); \
  ymm26 = _mm256_setzero_pd(); \
  ymm27 = _mm256_setzero_pd(); \
  ymm28 = _mm256_setzero_pd(); \
  ymm29 = _mm256_setzero_pd(); \
  ymm30 = _mm256_setzero_pd(); \
  ymm31 = _mm256_setzero_pd();

#define BLIS_SET_ZMM_REG_ZEROS \
  zmm0 = _mm512_setzero_pd(); \
  zmm1 = _mm512_setzero_pd(); \
  zmm2 = _mm512_setzero_pd(); \
  zmm3 = _mm512_setzero_pd(); \
  zmm4 = _mm512_setzero_pd(); \
  zmm5 = _mm512_setzero_pd(); \
  zmm6 = _mm512_setzero_pd(); \
  zmm7 = _mm512_setzero_pd(); \
  zmm8 = _mm512_setzero_pd(); \
  zmm9 = _mm512_setzero_pd(); \
  zmm10 = _mm512_setzero_pd(); \
  zmm11 = _mm512_setzero_pd(); \
  zmm12 = _mm512_setzero_pd(); \
  zmm13 = _mm512_setzero_pd(); \
  zmm14 = _mm512_setzero_pd(); \
  zmm15 = _mm512_setzero_pd(); \
  zmm16 = _mm512_setzero_pd(); \
  zmm17 = _mm512_setzero_pd(); \
  zmm18 = _mm512_setzero_pd(); \
  zmm19 = _mm512_setzero_pd(); \
  zmm20 = _mm512_setzero_pd(); \
  zmm21 = _mm512_setzero_pd(); \
  zmm22 = _mm512_setzero_pd(); \
  zmm23 = _mm512_setzero_pd(); \
  zmm24 = _mm512_setzero_pd(); \
  zmm25 = _mm512_setzero_pd(); \
  zmm26 = _mm512_setzero_pd(); \
  zmm27 = _mm512_setzero_pd(); \
  zmm28 = _mm512_setzero_pd(); \
  zmm29 = _mm512_setzero_pd(); \
  zmm30 = _mm512_setzero_pd(); \
  zmm31 = _mm512_setzero_pd();

#define BLIS_SET_YMM_REG_ZEROS_FOR_N_REM \
  ymm3 = _mm256_setzero_pd(); \
  ymm4 = _mm256_setzero_pd(); \
  ymm5 = _mm256_setzero_pd(); \
  ymm6 = _mm256_setzero_pd(); \
  ymm7 = _mm256_setzero_pd(); \
  ymm8 = _mm256_setzero_pd(); \
  ymm9 = _mm256_setzero_pd(); \
  ymm10 = _mm256_setzero_pd(); \
  ymm15 = _mm256_setzero_pd(); \

/*
   declaration of trsm small kernels function pointer
*/
typedef err_t (*trsmsmall_ker_ft)
     (
       obj_t*   AlphaObj,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     );

/*
  Pack a block of 8xk from input buffer into packed buffer
  directly or after transpose based on input params
*/
BLIS_INLINE void bli_dtrsm_small_pack_avx512
     (
       char     side,
       dim_t    size,
       bool     trans,
       double*  inbuf,
       dim_t    cs_a,
       double*  pbuff,
       dim_t    p_lda,
       dim_t    mr
     )
{
  // scratch registers
  __m512d zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  if (side == 'L' || side == 'l')
  {
    return; // BLIS_NOT_YET_IMPLEMENTED
  }
  else if (side == 'R' || side == 'r')
  {
    if (trans)
    {
      /*
          ----------------   -------------
          |           |      |     |     |
          |    4x8    |      |     |     |
          -------------  ==> | 8x4 | 8x4 |
          |    4x8    |      |     |     |
          |           |      |     |     |
          ----------------   -------------
      */
      __m256d ymm0, ymm1, ymm2, ymm3;
      __m256d ymm4, ymm5, ymm6, ymm7;
      __m256d ymm8, ymm9, ymm10, ymm11;
      __m256d ymm12, ymm13;
      for (dim_t x = 0; x < p_lda; x += mr)
      {
        // load 4x8
        ymm0 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 0)));
        ymm1 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 1)));
        ymm2 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 2)));
        ymm3 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 3)));
        ymm10 = _mm256_loadu_pd((double const *)(inbuf + 4 + (cs_a * 0)));
        ymm11 = _mm256_loadu_pd((double const *)(inbuf + 4 + (cs_a * 1)));
        ymm12 = _mm256_loadu_pd((double const *)(inbuf + 4 + (cs_a * 2)));
        ymm13 = _mm256_loadu_pd((double const *)(inbuf + 4 + (cs_a * 3)));

        // transpose 4x4
        ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
        ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
        ymm6 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);
        ymm8 = _mm256_permute2f128_pd(ymm4, ymm5, 0x31);
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
        ymm7 = _mm256_permute2f128_pd(ymm0, ymm1, 0x20);
        ymm9 = _mm256_permute2f128_pd(ymm0, ymm1, 0x31);

        // store 4x4
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 0)), ymm6);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 1)), ymm7);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 2)), ymm8);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 3)), ymm9);

        // transpose 4x4
        ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
        ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);
        ymm6 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);
        ymm8 = _mm256_permute2f128_pd(ymm4, ymm5, 0x31);
        ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
        ymm1 = _mm256_unpackhi_pd(ymm12, ymm13);
        ymm7 = _mm256_permute2f128_pd(ymm0, ymm1, 0x20);
        ymm9 = _mm256_permute2f128_pd(ymm0, ymm1, 0x31);

        // store 4x4
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 4)), ymm6);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 5)), ymm7);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 6)), ymm8);
        _mm256_storeu_pd((double *)(pbuff + (p_lda * 7)), ymm9);

        // load 4x8
        ymm0 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 4)));
        ymm1 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 5)));
        ymm2 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 6)));
        ymm3 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 7)));
        ymm10 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 4) + 4));
        ymm11 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 5) + 4));
        ymm12 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 6) + 4));
        ymm13 = _mm256_loadu_pd((double const *)(inbuf + (cs_a * 7) + 4));

        // transpose 4x4
        ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
        ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
        ymm6 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);
        ymm8 = _mm256_permute2f128_pd(ymm4, ymm5, 0x31);
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
        ymm7 = _mm256_permute2f128_pd(ymm0, ymm1, 0x20);
        ymm9 = _mm256_permute2f128_pd(ymm0, ymm1, 0x31);

        // store 4x4
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 0)), ymm6);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 1)), ymm7);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 2)), ymm8);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 3)), ymm9);

        // transpose 4x4
        ymm4 = _mm256_unpacklo_pd(ymm10, ymm11);
        ymm5 = _mm256_unpacklo_pd(ymm12, ymm13);
        ymm6 = _mm256_permute2f128_pd(ymm4, ymm5, 0x20);
        ymm8 = _mm256_permute2f128_pd(ymm4, ymm5, 0x31);
        ymm0 = _mm256_unpackhi_pd(ymm10, ymm11);
        ymm1 = _mm256_unpackhi_pd(ymm12, ymm13);
        ymm7 = _mm256_permute2f128_pd(ymm0, ymm1, 0x20);
        ymm9 = _mm256_permute2f128_pd(ymm0, ymm1, 0x31);

        // store 4x4
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 4)), ymm6);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 5)), ymm7);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 6)), ymm8);
        _mm256_storeu_pd((double *)(pbuff + 4 + (p_lda * 7)), ymm9);

        inbuf += mr * cs_a;
        pbuff += mr;
      }
    }
    else
    {
      dim_t size_div_8 = size >> 3;
      for (int i = 0; i < size_div_8; i++)
      {
        zmm0 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 0)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 0)), zmm0);
        zmm1 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 1)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 1)), zmm1);
        zmm2 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 2)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 2)), zmm2);
        zmm3 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 3)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 3)), zmm3);
        zmm4 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 4)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 4)), zmm4);
        zmm5 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 5)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 5)), zmm5);
        zmm6 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 6)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 6)), zmm6);
        zmm7 = _mm512_loadu_pd((double const *)(inbuf + (cs_a * 7)));
        _mm512_storeu_pd((double *)(pbuff + (p_lda * 7)), zmm7);
        inbuf += 8;
        pbuff += 8;
      }
    }
  }
}
/*
  Pack diagonal elements of A block (8) into an array
  a. This helps in utilze cache line efficiently in TRSM operation
  b. store ones when input is unit diagonal
*/
BLIS_INLINE void dtrsm_small_pack_diag_element_avx512
     (
       bool     is_unitdiag,
       double*  a11,
       dim_t    cs_a,
       double*  d11_pack,
       dim_t    size
     )
{
  __m512d zmm0, zmm1, zmm2, zmm3;
  __m512d zmm4, zmm5, zmm6, zmm7;
  __m512d zmm8;
  double ones = 1.0;
  // if (size == 8)
  {
    zmm8 = _mm512_set1_pd(ones);
    if (!is_unitdiag)
    {
      __m512d zmm10, zmm11, zmm12, zmm13;
      __m512d zmm14, zmm15;
      // broadcast diagonal elements of A11
      zmm0 = _mm512_set1_pd(*(a11 + (cs_a * 0) + 0));
      zmm1 = _mm512_set1_pd(*(a11 + (cs_a * 1) + 1));
      zmm2 = _mm512_set1_pd(*(a11 + (cs_a * 2) + 2));
      zmm3 = _mm512_set1_pd(*(a11 + (cs_a * 3) + 3));
      zmm4 = _mm512_set1_pd(*(a11 + (cs_a * 4) + 4));
      zmm5 = _mm512_set1_pd(*(a11 + (cs_a * 5) + 5));
      zmm6 = _mm512_set1_pd(*(a11 + (cs_a * 6) + 6));
      zmm7 = _mm512_set1_pd(*(a11 + (cs_a * 7) + 7));

      //combine all elements of A11 into zmm1
      // Stage 1
      zmm10 = _mm512_unpacklo_pd(zmm0, zmm1);
      zmm11 = _mm512_unpacklo_pd(zmm2, zmm3);
      zmm12 = _mm512_unpacklo_pd(zmm4, zmm5);
      zmm13 = _mm512_unpacklo_pd(zmm6, zmm7);
      // Stage 2
      zmm14 = _mm512_shuffle_f64x2(zmm10, zmm11, 0b10001000);
      zmm15 = _mm512_shuffle_f64x2(zmm12, zmm13, 0b10001000);
      // Stage 3
      zmm1 = _mm512_shuffle_f64x2(zmm14, zmm15, 0b10001000);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
        zmm8 = zmm1;
      #endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
        zmm8 = _mm512_div_pd(zmm8, zmm1);
      #endif
    }
    _mm512_storeu_pd((double *)(d11_pack), zmm8);
  }
}
/*
 * Kernels Table
 */
trsmsmall_ker_ft ker_fps_AVX512[4][8] =
  {
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
    {bli_dtrsm_small_AutXB_AlXB_AVX512,
     bli_dtrsm_small_AltXB_AuXB_AVX512,
     bli_dtrsm_small_AltXB_AuXB_AVX512,
     bli_dtrsm_small_AutXB_AlXB_AVX512,
     bli_dtrsm_small_XAutB_XAlB_AVX512,
     bli_dtrsm_small_XAltB_XAuB_AVX512,
     bli_dtrsm_small_XAltB_XAuB_AVX512,
     bli_dtrsm_small_XAutB_XAlB_AVX512},
    {NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL,
     NULL},
};
/*
* The bli_trsm_small implements a version of TRSM where A is packed and reused
*
* Input:  A: MxM (triangular matrix)
*     B: MxN matrix
* Output: X: MxN matrix such that
       AX = alpha*B or XA = alpha*B or A'X = alpha*B or XA' = alpha*B
* Here the output X is stored in B
*
* Note: Currently only dtrsm is supported when A & B are column-major
*/
err_t bli_trsm_small_AVX512
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl,
       bool     is_parallel
     )
{
  err_t err;
  dim_t m = bli_obj_length(b);
  dim_t n = bli_obj_width(b);

  bool uplo = bli_obj_is_upper(a);
  bool transa = bli_obj_has_trans(a);
  num_t dt = bli_obj_dt(a);

  switch (dt)
  {
  case BLIS_DOUBLE:
  {
    if ((!is_parallel) && (m > 1200 || n > 1200))
    {
      return BLIS_NOT_YET_IMPLEMENTED;
    }
    break;
  }
  case BLIS_FLOAT:
  case BLIS_SCOMPLEX:
  case BLIS_DCOMPLEX:
  default:
  {
    return BLIS_NOT_YET_IMPLEMENTED;
    break;
  }
  }
  /* If alpha is zero, B matrix will become zero after scaling
     hence solution is also zero matrix */
  if (bli_obj_equals(alpha, &BLIS_ZERO))
  {
    return BLIS_NOT_YET_IMPLEMENTED; // scale B by alpha
  }

  // Return if inputs are row major as currently
  // we are supporing col major only
  if ((bli_obj_row_stride(a) != 1) ||
    (bli_obj_row_stride(b) != 1))
  {
    return BLIS_INVALID_ROW_STRIDE;
  }

  // A is expected to be triangular in trsm
  if (!bli_obj_is_upper_or_lower(a))
  {
    return BLIS_EXPECTED_TRIANGULAR_OBJECT;
  }
  /*
   *  Compose kernel index based on inputs
   */
  dim_t keridx = (((side & 0x1) << 2) |
          ((uplo & 0x1) << 1) |
          (transa & 0x1));
  trsmsmall_ker_ft ker_fp = ker_fps_AVX512[dt][keridx];
  /*Call the kernel*/
  err = ker_fp(
    alpha,
    a,
    b,
    cntx,
    cntl);
  return err;
};

#ifdef BLIS_ENABLE_OPENMP
/*
 * Parallelized dtrsm_small across m-dimension or n-dimension based on side(Left/Right)
 */
err_t bli_trsm_small_mt_AVX512
     (
       side_t   side,
       obj_t*   alpha,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     )
{
  gint_t m = bli_obj_length(b); // number of rows of matrix b
  gint_t n = bli_obj_width(b);  // number of columns of Matrix b
  dim_t d_mr = 8,d_nr = 8;

  num_t dt = bli_obj_dt(a);
  switch (dt)
  {
    case BLIS_DOUBLE:
    {
      d_mr = 8, d_nr = 8;
      break;
    }
    default:
    {
      return BLIS_NOT_YET_IMPLEMENTED;
      break;
    }
  }

  rntm_t rntm;
  bli_rntm_init_from_global(&rntm);

#ifdef AOCL_DYNAMIC
  // If dynamic-threading is enabled, calculate optimum number
  //  of threads.
  //  rntm will be updated with optimum number of threads.
  if (bli_obj_is_double(b))
  {
    bli_nthreads_optimum(a, b, b, BLIS_TRSM, &rntm);
  }
#endif

  // Query the total number of threads from the rntm_t object.
  dim_t n_threads = bli_rntm_num_threads(&rntm);

  if (n_threads < 0)
    n_threads = 1;

  bool is_parallel = bli_thread_get_is_parallel();

  err_t status = BLIS_SUCCESS;
  _Pragma("omp parallel num_threads(n_threads)")
  {
    // Query the thread's id from OpenMP.
    const dim_t tid = omp_get_thread_num();
    const dim_t nt_real = omp_get_num_threads();

    // if num threads requested and num thread available
    // is not same then use single thread small
    if(nt_real != n_threads)
    {
      if(tid == 0)
      {
        bli_trsm_small
            (
              side,
              alpha,
              a,
              b,
              cntx,
              cntl,
              is_parallel
            );
      }
    }
    else
    {
      obj_t b_t;
      dim_t start; // Each thread start Index
      dim_t end;   // Each thread end Index
      thrinfo_t thread;

      thread.n_way = n_threads;
      thread.work_id = tid;
      thread.ocomm_id = tid;

      // Compute start and end indexes of matrix partitioning for each thread
      if (bli_is_right(side))
      {
        bli_thread_range_sub
                (
                  &thread,
                  m,
                  d_mr, // Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
        // For each thread acquire matrix block on which they operate
        // Data-based parallelism

        bli_acquire_mpart_mdim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
      }
      else
      {
        bli_thread_range_sub
                (
                  &thread,
                  n,
                  d_nr,// Need to decide based on type
                  FALSE,
                  &start,
                  &end
                );
        // For each thread acquire matrix block on which they operate
        // Data-based parallelism

        bli_acquire_mpart_ndim(BLIS_FWD, BLIS_SUBPART1, start, end - start, b, &b_t);
      }

      // Parallelism is only across m-dimension/n-dimension - therefore matrix a is common to
      // all threads
      err_t status_l = BLIS_SUCCESS;

      status_l = bli_trsm_small_AVX512
              (
                side,
                alpha,
                a,
                &b_t,
                NULL,
                NULL,
                is_parallel
              );
      // To capture the error populated from any of the threads
      _Pragma("omp critical")
      status = (status != BLIS_NOT_YET_IMPLEMENTED) ? status_l : status;
    }
  }

  return status;
} // End of function
#endif

// region - GEMM DTRSM for right variants

#define BLIS_DTRSM_SMALL_GEMM_8nx8m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  /*K loop is broken into two seperate loops
    each loop computes k/2 iterations */ \
  \
  int itr = (k_iter / 2); /*itr count for first loop*/\
  int itr2 = k_iter - itr; /*itr count for second loop*/\
  double *a01_2 = a01 + itr; /*a01 for second loop*/\
  double *b10_2 = b10 + (cs_b * itr); /*b10 for second loop*/\
  for (; itr > 0; itr--) \
  { \
    zmm0 = _mm512_loadu_pd((double const *)b10); \
    \
    zmm1 = _mm512_set1_pd(*(a01 + (p_lda * 0))); \
    zmm2 = _mm512_set1_pd(*(a01 + (p_lda * 1))); \
    zmm3 = _mm512_set1_pd(*(a01 + (p_lda * 2))); \
    zmm4 = _mm512_set1_pd(*(a01 + (p_lda * 3))); \
    zmm5 = _mm512_set1_pd(*(a01 + (p_lda * 4))); \
    zmm6 = _mm512_set1_pd(*(a01 + (p_lda * 5))); \
    zmm7 = _mm512_set1_pd(*(a01 + (p_lda * 6))); \
    zmm8 = _mm512_set1_pd(*(a01 + (p_lda * 7))); \
    \
    /*prefetch b10 4 iterations in advance*/ \
    _mm_prefetch((b10 + 4 * cs_b), _MM_HINT_T0); \
    zmm9  = _mm512_fmadd_pd(zmm1, zmm0, zmm9 ); \
    zmm10 = _mm512_fmadd_pd(zmm2, zmm0, zmm10); \
    zmm11 = _mm512_fmadd_pd(zmm3, zmm0, zmm11); \
    zmm12 = _mm512_fmadd_pd(zmm4, zmm0, zmm12); \
    zmm13 = _mm512_fmadd_pd(zmm5, zmm0, zmm13); \
    zmm14 = _mm512_fmadd_pd(zmm6, zmm0, zmm14); \
    zmm15 = _mm512_fmadd_pd(zmm7, zmm0, zmm15); \
    zmm16 = _mm512_fmadd_pd(zmm8, zmm0, zmm16); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  } \
  for (; itr2 > 0; itr2--) \
  { \
    zmm23 = _mm512_loadu_pd((double const *)b10_2); \
    \
    zmm17 = _mm512_set1_pd(*(a01_2 + (p_lda * 0))); \
    zmm18 = _mm512_set1_pd(*(a01_2 + (p_lda * 1))); \
    zmm19 = _mm512_set1_pd(*(a01_2 + (p_lda * 2))); \
    zmm20 = _mm512_set1_pd(*(a01_2 + (p_lda * 3))); \
    zmm21 = _mm512_set1_pd(*(a01_2 + (p_lda * 4))); \
    zmm22 = _mm512_set1_pd(*(a01_2 + (p_lda * 5))); \
    \
    _mm_prefetch((b10_2 + 4 * cs_b), _MM_HINT_T0); \
    zmm24 = _mm512_fmadd_pd(zmm17, zmm23, zmm24); \
    zmm17 = _mm512_set1_pd(*(a01_2 + (p_lda * 6))); \
    zmm25 = _mm512_fmadd_pd(zmm18, zmm23, zmm25); \
    zmm18 = _mm512_set1_pd(*(a01_2 + (p_lda * 7))); \
    zmm26 = _mm512_fmadd_pd(zmm19, zmm23, zmm26); \
    zmm27 = _mm512_fmadd_pd(zmm20, zmm23, zmm27); \
    zmm28 = _mm512_fmadd_pd(zmm21, zmm23, zmm28); \
    zmm29 = _mm512_fmadd_pd(zmm22, zmm23, zmm29); \
    zmm30 = _mm512_fmadd_pd(zmm17, zmm23, zmm30); \
    zmm31 = _mm512_fmadd_pd(zmm18, zmm23, zmm31); \
    \
    a01_2 += 1; \
    b10_2 += cs_b; \
  } \
  \
  /*prefetch 8 columns of b11)*/ \
  _mm_prefetch((b11 + (0) * cs_b), _MM_HINT_T0); \
  /*combine the results of both loops*/ \
  zmm9 = _mm512_add_pd(zmm9, zmm24); \
  _mm_prefetch((b11 + (1) * cs_b), _MM_HINT_T0); \
  zmm10 = _mm512_add_pd(zmm10, zmm25); \
  _mm_prefetch((b11 + (2) * cs_b), _MM_HINT_T0); \
  zmm11 = _mm512_add_pd(zmm11, zmm26); \
  _mm_prefetch((b11 + (3) * cs_b), _MM_HINT_T0); \
  zmm12 = _mm512_add_pd(zmm12, zmm27); \
  _mm_prefetch((b11 + (4) * cs_b), _MM_HINT_T0); \
  zmm13 = _mm512_add_pd(zmm13, zmm28); \
  _mm_prefetch((b11 + (5) * cs_b), _MM_HINT_T0); \
  zmm14 = _mm512_add_pd(zmm14, zmm29); \
  _mm_prefetch((b11 + (6) * cs_b), _MM_HINT_T0); \
  zmm15 = _mm512_add_pd(zmm15, zmm30); \
  _mm_prefetch((b11 + (7) * cs_b), _MM_HINT_T0); \
  zmm16 = _mm512_add_pd(zmm16, zmm31);
/*
// alternative way to prrefetch b11
//  itr2 = itr2 + itr + 8; \
//  for(;itr2>0;itr2--) \
//   {\
//   zmm23 = _mm512_loadu_pd((double const *)b10_2); \
//   \
//   zmm17 = _mm512_set1_pd(*(a01_2 + p_lda * 0)); \
//   zmm18 = _mm512_set1_pd(*(a01_2 + p_lda * 1)); \
//   zmm19 = _mm512_set1_pd(*(a01_2 + p_lda * 2)); \
//   zmm20 = _mm512_set1_pd(*(a01_2 + p_lda * 3)); \
//   zmm21 = _mm512_set1_pd(*(a01_2 + p_lda * 4)); \
//   zmm22 = _mm512_set1_pd(*(a01_2 + p_lda * 5)); \
//   \
//   _mm_prefetch((b10_2 + 4*cs_b), _MM_HINT_T0); \
//   _mm_prefetch((b11 + (itr2-1)*cs_b), _MM_HINT_T0); \
//   zmm24 = _mm512_fmadd_pd(zmm17, zmm23, zmm24); \
//   zmm17 = _mm512_set1_pd(*(a01_2 + p_lda * 6)); \
//   zmm25 = _mm512_fmadd_pd(zmm18, zmm23, zmm25); \
//   zmm18 = _mm512_set1_pd(*(a01_2 + p_lda * 7)); \
//   zmm26 = _mm512_fmadd_pd(zmm19, zmm23, zmm26); \
//   zmm27 = _mm512_fmadd_pd(zmm20, zmm23, zmm27); \
//   zmm28 = _mm512_fmadd_pd(zmm21, zmm23, zmm28); \
//   zmm29 = _mm512_fmadd_pd(zmm22, zmm23, zmm29); \
//   zmm30 = _mm512_fmadd_pd(zmm17, zmm23, zmm30); \
//   zmm31 = _mm512_fmadd_pd(zmm18, zmm23, zmm31); \
//   \
//   a01_2 += 1;\
//   b10_2 += cs_b; \
//   }\
*/
/*
// alternative version of main loop
#define BLIS_DTRSM_SMALL_GEMM_8nx8m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  int itr = k_iter - 8; \
  for(;itr>0;itr--) \
  {\
  zmm0 = _mm512_loadu_pd((double const *)b10); \
  \
  zmm1 = _mm512_set1_pd(*(a01 + p_lda * 0)); \
  zmm2 = _mm512_set1_pd(*(a01 + p_lda * 1)); \
  zmm3 = _mm512_set1_pd(*(a01 + p_lda * 2)); \
  zmm4 = _mm512_set1_pd(*(a01 + p_lda * 3)); \
  zmm5 = _mm512_set1_pd(*(a01 + p_lda * 4)); \
  zmm6 = _mm512_set1_pd(*(a01 + p_lda * 5)); \
  zmm7 = _mm512_set1_pd(*(a01 + p_lda * 6)); \
  zmm8 = _mm512_set1_pd(*(a01 + p_lda * 7)); \
  \
  _mm_prefetch((b10 + 4*cs_b), _MM_HINT_T0); \
  zmm9  = _mm512_fmadd_pd(zmm1, zmm0, zmm9 ); \
  zmm10 = _mm512_fmadd_pd(zmm2, zmm0, zmm10); \
  zmm11 = _mm512_fmadd_pd(zmm3, zmm0, zmm11); \
  zmm12 = _mm512_fmadd_pd(zmm4, zmm0, zmm12); \
  zmm13 = _mm512_fmadd_pd(zmm5, zmm0, zmm13); \
  zmm14 = _mm512_fmadd_pd(zmm6, zmm0, zmm14); \
  zmm15 = _mm512_fmadd_pd(zmm7, zmm0, zmm15); \
  zmm16 = _mm512_fmadd_pd(zmm8, zmm0, zmm16); \
  \
  a01 += 1;\
  b10 += cs_b; \
  }\
  itr += 8; \
  for(;itr>0;itr--) \
  {\
  zmm0 = _mm512_loadu_pd((double const *)b10); \
  \
  zmm1 = _mm512_set1_pd(*(a01 + p_lda * 0)); \
  zmm2 = _mm512_set1_pd(*(a01 + p_lda * 1)); \
  zmm3 = _mm512_set1_pd(*(a01 + p_lda * 2)); \
  zmm4 = _mm512_set1_pd(*(a01 + p_lda * 3)); \
  zmm5 = _mm512_set1_pd(*(a01 + p_lda * 4)); \
  zmm6 = _mm512_set1_pd(*(a01 + p_lda * 5)); \
  zmm7 = _mm512_set1_pd(*(a01 + p_lda * 6)); \
  zmm8 = _mm512_set1_pd(*(a01 + p_lda * 7)); \
  \
  _mm_prefetch((b10 + 4*cs_b), _MM_HINT_T0); \
  _mm_prefetch((b11 + (itr-1)*cs_b), _MM_HINT_T0); \
  zmm9  = _mm512_fmadd_pd(zmm1, zmm0, zmm9 ); \
  zmm10 = _mm512_fmadd_pd(zmm2, zmm0, zmm10); \
  zmm11 = _mm512_fmadd_pd(zmm3, zmm0, zmm11); \
  zmm12 = _mm512_fmadd_pd(zmm4, zmm0, zmm12); \
  zmm13 = _mm512_fmadd_pd(zmm5, zmm0, zmm13); \
  zmm14 = _mm512_fmadd_pd(zmm6, zmm0, zmm14); \
  zmm15 = _mm512_fmadd_pd(zmm7, zmm0, zmm15); \
  zmm16 = _mm512_fmadd_pd(zmm8, zmm0, zmm16); \
  \
  a01 += 1;\
  b10 += cs_b; \
  }\
*/

#define BLIS_DTRSM_SMALL_GEMM_8nx4m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  /*K loop is broken into two seperate loops
    each loop computes k/2 iterations */ \
  \
  int itr = (k_iter / 2); /*itr count for first loop*/\
  int itr2 = k_iter - itr; /*itr count for second loop*/\
  double *a01_2 = a01 + itr; /*a01 for second loop*/\
  double *b10_2 = b10 + (cs_b * itr); /*b10 for second loop*/\
  for (; itr > 0; itr--) \
  { \
    ymm0 = _mm256_loadu_pd((double const *)(b10)); \
    \
    ymm1 = _mm256_broadcast_sd((a01 + (p_lda * 0))); \
    ymm2 = _mm256_broadcast_sd((a01 + (p_lda * 1))); \
    ymm3 = _mm256_broadcast_sd((a01 + (p_lda * 2))); \
    ymm4 = _mm256_broadcast_sd((a01 + (p_lda * 3))); \
    ymm5 = _mm256_broadcast_sd((a01 + (p_lda * 4))); \
    ymm6 = _mm256_broadcast_sd((a01 + (p_lda * 5))); \
    ymm7 = _mm256_broadcast_sd((a01 + (p_lda * 6))); \
    ymm8 = _mm256_broadcast_sd((a01 + (p_lda * 7))); \
    \
    _mm_prefetch((b10 + 4 * cs_b), _MM_HINT_T0); \
    ymm9  = _mm256_fmadd_pd(ymm1, ymm0, ymm9 ); \
    ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10); \
    ymm11 = _mm256_fmadd_pd(ymm3, ymm0, ymm11); \
    ymm12 = _mm256_fmadd_pd(ymm4, ymm0, ymm12); \
    ymm13 = _mm256_fmadd_pd(ymm5, ymm0, ymm13); \
    ymm14 = _mm256_fmadd_pd(ymm6, ymm0, ymm14); \
    ymm15 = _mm256_fmadd_pd(ymm7, ymm0, ymm15); \
    ymm16 = _mm256_fmadd_pd(ymm8, ymm0, ymm16); \
    \
    a01 += 1; \
    b10 += cs_b; \
  } \
  for (; itr2 > 0; itr2--) \
  { \
    ymm23 = _mm256_loadu_pd((double const *)(b10_2)); \
    \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 0))); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 1))); \
    ymm19 = _mm256_broadcast_sd((a01_2 + (p_lda * 2))); \
    ymm20 = _mm256_broadcast_sd((a01_2 + (p_lda * 3))); \
    ymm21 = _mm256_broadcast_sd((a01_2 + (p_lda * 4))); \
    ymm22 = _mm256_broadcast_sd((a01_2 + (p_lda * 5))); \
    \
    _mm_prefetch((b10_2 + 4 * cs_b), _MM_HINT_T0); \
    ymm24 = _mm256_fmadd_pd(ymm17, ymm23, ymm24); \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 6))); \
    ymm25 = _mm256_fmadd_pd(ymm18, ymm23, ymm25); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 7))); \
    ymm26 = _mm256_fmadd_pd(ymm19, ymm23, ymm26); \
    ymm27 = _mm256_fmadd_pd(ymm20, ymm23, ymm27); \
    ymm28 = _mm256_fmadd_pd(ymm21, ymm23, ymm28); \
    ymm29 = _mm256_fmadd_pd(ymm22, ymm23, ymm29); \
    ymm30 = _mm256_fmadd_pd(ymm17, ymm23, ymm30); \
    ymm31 = _mm256_fmadd_pd(ymm18, ymm23, ymm31); \
    \
    a01_2 += 1; \
    b10_2 += cs_b; \
  } \
  /*combine the results of both loops*/ \
  _mm_prefetch((b11 + (0) * cs_b), _MM_HINT_T0); \
  ymm9  = _mm256_add_pd(ymm9, ymm24); \
  _mm_prefetch((b11 + (1) * cs_b), _MM_HINT_T0); \
  ymm10 = _mm256_add_pd(ymm10, ymm25); \
  _mm_prefetch((b11 + (2) * cs_b), _MM_HINT_T0); \
  ymm11 = _mm256_add_pd(ymm11, ymm26); \
  _mm_prefetch((b11 + (3) * cs_b), _MM_HINT_T0); \
  ymm12 = _mm256_add_pd(ymm12, ymm27); \
  _mm_prefetch((b11 + (4) * cs_b), _MM_HINT_T0); \
  ymm13 = _mm256_add_pd(ymm13, ymm28); \
  _mm_prefetch((b11 + (5) * cs_b), _MM_HINT_T0); \
  ymm14 = _mm256_add_pd(ymm14, ymm29); \
  _mm_prefetch((b11 + (6) * cs_b), _MM_HINT_T0); \
  ymm15 = _mm256_add_pd(ymm15, ymm30); \
  _mm_prefetch((b11 + (7) * cs_b), _MM_HINT_T0); \
  ymm16 = _mm256_add_pd(ymm16, ymm31);


#define BLIS_DTRSM_SMALL_GEMM_8nx3m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  /*K loop is broken into two seperate loops
    each loop computes k/2 iterations */ \
  \
  int itr = (k_iter / 2); /*itr count for first loop*/\
  int itr2 = k_iter - itr; /*itr count for second loop*/\
  double *a01_2 = a01 + itr; /*a01 for second loop*/\
  double *b10_2 = b10 + (cs_b * itr); /*b10 for second loop*/\
  for (; itr > 0; itr--) \
  { \
    xmm5 = _mm_loadu_pd((b10)); /*load b10[0] and b10[1] into xmm5*/\
    ymm0 = _mm256_broadcast_sd((b10 + 2)); /*broadcast b10[2] into ymm0*/\
    ymm0 = _mm256_insertf64x2(ymm0, xmm5, 0); \
    /*ymm0 = {b10[0], b10[1], b10[2], b10[2]}*/\
    \
    ymm1 = _mm256_broadcast_sd((a01 + (p_lda * 0))); \
    ymm2 = _mm256_broadcast_sd((a01 + (p_lda * 1))); \
    ymm3 = _mm256_broadcast_sd((a01 + (p_lda * 2))); \
    ymm4 = _mm256_broadcast_sd((a01 + (p_lda * 3))); \
    ymm5 = _mm256_broadcast_sd((a01 + (p_lda * 4))); \
    ymm6 = _mm256_broadcast_sd((a01 + (p_lda * 5))); \
    ymm7 = _mm256_broadcast_sd((a01 + (p_lda * 6))); \
    ymm8 = _mm256_broadcast_sd((a01 + (p_lda * 7))); \
    \
    _mm_prefetch((b10 + 4 * cs_b), _MM_HINT_T0); \
    ymm9  = _mm256_fmadd_pd(ymm1, ymm0, ymm9 ); \
    ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10); \
    ymm11 = _mm256_fmadd_pd(ymm3, ymm0, ymm11); \
    ymm12 = _mm256_fmadd_pd(ymm4, ymm0, ymm12); \
    ymm13 = _mm256_fmadd_pd(ymm5, ymm0, ymm13); \
    ymm14 = _mm256_fmadd_pd(ymm6, ymm0, ymm14); \
    ymm15 = _mm256_fmadd_pd(ymm7, ymm0, ymm15); \
    ymm16 = _mm256_fmadd_pd(ymm8, ymm0, ymm16); \
    \
    a01 += 1; \
    b10 += cs_b; \
  } \
  for (; itr2 > 0; itr2--) \
  { \
    xmm0 = _mm_loadu_pd((b10_2)); \
    ymm23 = _mm256_broadcast_sd((b10_2 + 2)); \
    ymm23 = _mm256_insertf64x2(ymm23, xmm0, 0); \
    \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 0))); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 1))); \
    ymm19 = _mm256_broadcast_sd((a01_2 + (p_lda * 2))); \
    ymm20 = _mm256_broadcast_sd((a01_2 + (p_lda * 3))); \
    ymm21 = _mm256_broadcast_sd((a01_2 + (p_lda * 4))); \
    ymm22 = _mm256_broadcast_sd((a01_2 + (p_lda * 5))); \
    \
    _mm_prefetch((b10_2 + 4 * cs_b), _MM_HINT_T0); \
    ymm24 = _mm256_fmadd_pd(ymm17, ymm23, ymm24); \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 6))); \
    ymm25 = _mm256_fmadd_pd(ymm18, ymm23, ymm25); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 7))); \
    ymm26 = _mm256_fmadd_pd(ymm19, ymm23, ymm26); \
    ymm27 = _mm256_fmadd_pd(ymm20, ymm23, ymm27); \
    ymm28 = _mm256_fmadd_pd(ymm21, ymm23, ymm28); \
    ymm29 = _mm256_fmadd_pd(ymm22, ymm23, ymm29); \
    ymm30 = _mm256_fmadd_pd(ymm17, ymm23, ymm30); \
    ymm31 = _mm256_fmadd_pd(ymm18, ymm23, ymm31); \
    \
    a01_2 += 1; \
    b10_2 += cs_b; \
  } \
  /*combine the results of both loops*/ \
  _mm_prefetch((b11 + (0) * cs_b), _MM_HINT_T0); \
  ymm9  = _mm256_add_pd(ymm9, ymm24); \
  _mm_prefetch((b11 + (1) * cs_b), _MM_HINT_T0); \
  ymm10 = _mm256_add_pd(ymm10, ymm25); \
  _mm_prefetch((b11 + (2) * cs_b), _MM_HINT_T0); \
  ymm11 = _mm256_add_pd(ymm11, ymm26); \
  _mm_prefetch((b11 + (3) * cs_b), _MM_HINT_T0); \
  ymm12 = _mm256_add_pd(ymm12, ymm27); \
  _mm_prefetch((b11 + (4) * cs_b), _MM_HINT_T0); \
  ymm13 = _mm256_add_pd(ymm13, ymm28); \
  _mm_prefetch((b11 + (5) * cs_b), _MM_HINT_T0); \
  ymm14 = _mm256_add_pd(ymm14, ymm29); \
  _mm_prefetch((b11 + (6) * cs_b), _MM_HINT_T0); \
  ymm15 = _mm256_add_pd(ymm15, ymm30); \
  _mm_prefetch((b11 + (7) * cs_b), _MM_HINT_T0); \
  ymm16 = _mm256_add_pd(ymm16, ymm31);

  #define BLIS_DTRSM_SMALL_GEMM_8nx2m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  /*K loop is broken into two seperate loops
    each loop computes k/2 iterations */ \
  \
  int itr = (k_iter / 2); /*itr count for first loop*/\
  int itr2 = k_iter - itr; /*itr count for second loop*/\
  double *a01_2 = a01 + itr; /*a01 for second loop*/\
  double *b10_2 = b10 + (cs_b * itr); /*b10 for second loop*/\
  for (; itr > 0; itr--) \
  { \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_insertf64x2(ymm0, xmm5, 0); \
    \
    ymm1 = _mm256_broadcast_sd((a01 + (p_lda * 0))); \
    ymm2 = _mm256_broadcast_sd((a01 + (p_lda * 1))); \
    ymm3 = _mm256_broadcast_sd((a01 + (p_lda * 2))); \
    ymm4 = _mm256_broadcast_sd((a01 + (p_lda * 3))); \
    ymm5 = _mm256_broadcast_sd((a01 + (p_lda * 4))); \
    ymm6 = _mm256_broadcast_sd((a01 + (p_lda * 5))); \
    ymm7 = _mm256_broadcast_sd((a01 + (p_lda * 6))); \
    ymm8 = _mm256_broadcast_sd((a01 + (p_lda * 7))); \
    \
    _mm_prefetch((b10 + 4 * cs_b), _MM_HINT_T0); \
    ymm9  = _mm256_fmadd_pd(ymm1, ymm0, ymm9 ); \
    ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10); \
    ymm11 = _mm256_fmadd_pd(ymm3, ymm0, ymm11); \
    ymm12 = _mm256_fmadd_pd(ymm4, ymm0, ymm12); \
    ymm13 = _mm256_fmadd_pd(ymm5, ymm0, ymm13); \
    ymm14 = _mm256_fmadd_pd(ymm6, ymm0, ymm14); \
    ymm15 = _mm256_fmadd_pd(ymm7, ymm0, ymm15); \
    ymm16 = _mm256_fmadd_pd(ymm8, ymm0, ymm16); \
    \
    a01 += 1; \
    b10 += cs_b; \
  } \
  for (; itr2 > 0; itr2--) \
  { \
    xmm0 = _mm_loadu_pd((double const *)(b10_2)); \
    ymm23 = _mm256_insertf64x2(ymm23, xmm0, 0); \
    \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 0))); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 1))); \
    ymm19 = _mm256_broadcast_sd((a01_2 + (p_lda * 2))); \
    ymm20 = _mm256_broadcast_sd((a01_2 + (p_lda * 3))); \
    ymm21 = _mm256_broadcast_sd((a01_2 + (p_lda * 4))); \
    ymm22 = _mm256_broadcast_sd((a01_2 + (p_lda * 5))); \
    \
    _mm_prefetch((b10_2 + 4 * cs_b), _MM_HINT_T0); \
    ymm24 = _mm256_fmadd_pd(ymm17, ymm23, ymm24); \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 6))); \
    ymm25 = _mm256_fmadd_pd(ymm18, ymm23, ymm25); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 7))); \
    ymm26 = _mm256_fmadd_pd(ymm19, ymm23, ymm26); \
    ymm27 = _mm256_fmadd_pd(ymm20, ymm23, ymm27); \
    ymm28 = _mm256_fmadd_pd(ymm21, ymm23, ymm28); \
    ymm29 = _mm256_fmadd_pd(ymm22, ymm23, ymm29); \
    ymm30 = _mm256_fmadd_pd(ymm17, ymm23, ymm30); \
    ymm31 = _mm256_fmadd_pd(ymm18, ymm23, ymm31); \
    \
    a01_2 += 1; \
    b10_2 += cs_b; \
  } \
  /*combine the results of both loops*/ \
  _mm_prefetch((b11 + (0) * cs_b), _MM_HINT_T0); \
  ymm9 = _mm256_add_pd(ymm9, ymm24); \
  _mm_prefetch((b11 + (1) * cs_b), _MM_HINT_T0); \
  ymm10 = _mm256_add_pd(ymm10, ymm25); \
  _mm_prefetch((b11 + (2) * cs_b), _MM_HINT_T0); \
  ymm11 = _mm256_add_pd(ymm11, ymm26); \
  _mm_prefetch((b11 + (3) * cs_b), _MM_HINT_T0); \
  ymm12 = _mm256_add_pd(ymm12, ymm27); \
  _mm_prefetch((b11 + (4) * cs_b), _MM_HINT_T0); \
  ymm13 = _mm256_add_pd(ymm13, ymm28); \
  _mm_prefetch((b11 + (5) * cs_b), _MM_HINT_T0); \
  ymm14 = _mm256_add_pd(ymm14, ymm29); \
  _mm_prefetch((b11 + (6) * cs_b), _MM_HINT_T0); \
  ymm15 = _mm256_add_pd(ymm15, ymm30); \
  _mm_prefetch((b11 + (7) * cs_b), _MM_HINT_T0); \
  ymm16 = _mm256_add_pd(ymm16, ymm31);

#define BLIS_DTRSM_SMALL_GEMM_8nx1m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11) \
  /*K loop is broken into two seperate loops
    each loop computes k/2 iterations */ \
  \
  int itr = (k_iter / 2); /*itr count for first loop*/\
  int itr2 = k_iter - itr; /*itr count for second loop*/\
  double *a01_2 = a01 + itr; /*a01 for second loop*/\
  double *b10_2 = b10 + (cs_b * itr); /*b10 for second loop*/\
  for (; itr > 0; itr--) \
  { \
    ymm0 = _mm256_broadcast_sd(b10); \
    \
    ymm1 = _mm256_broadcast_sd((a01 + (p_lda * 0))); \
    ymm2 = _mm256_broadcast_sd((a01 + (p_lda * 1))); \
    ymm3 = _mm256_broadcast_sd((a01 + (p_lda * 2))); \
    ymm4 = _mm256_broadcast_sd((a01 + (p_lda * 3))); \
    ymm5 = _mm256_broadcast_sd((a01 + (p_lda * 4))); \
    ymm6 = _mm256_broadcast_sd((a01 + (p_lda * 5))); \
    ymm7 = _mm256_broadcast_sd((a01 + (p_lda * 6))); \
    ymm8 = _mm256_broadcast_sd((a01 + (p_lda * 7))); \
    \
    _mm_prefetch((b10 + 4 * cs_b), _MM_HINT_T0); \
    ymm9  = _mm256_fmadd_pd(ymm1, ymm0, ymm9 ); \
    ymm10 = _mm256_fmadd_pd(ymm2, ymm0, ymm10); \
    ymm11 = _mm256_fmadd_pd(ymm3, ymm0, ymm11); \
    ymm12 = _mm256_fmadd_pd(ymm4, ymm0, ymm12); \
    ymm13 = _mm256_fmadd_pd(ymm5, ymm0, ymm13); \
    ymm14 = _mm256_fmadd_pd(ymm6, ymm0, ymm14); \
    ymm15 = _mm256_fmadd_pd(ymm7, ymm0, ymm15); \
    ymm16 = _mm256_fmadd_pd(ymm8, ymm0, ymm16); \
    \
    a01 += 1; \
    b10 += cs_b; \
  } \
  for (; itr2 > 0; itr2--) \
  { \
    ymm23 = _mm256_broadcast_sd(b10_2); \
    \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 0))); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 1))); \
    ymm19 = _mm256_broadcast_sd((a01_2 + (p_lda * 2))); \
    ymm20 = _mm256_broadcast_sd((a01_2 + (p_lda * 3))); \
    ymm21 = _mm256_broadcast_sd((a01_2 + (p_lda * 4))); \
    ymm22 = _mm256_broadcast_sd((a01_2 + (p_lda * 5))); \
    \
    _mm_prefetch((b10_2 + 4 * cs_b), _MM_HINT_T0); \
    ymm24 = _mm256_fmadd_pd(ymm17, ymm23, ymm24); \
    ymm17 = _mm256_broadcast_sd((a01_2 + (p_lda * 6))); \
    ymm25 = _mm256_fmadd_pd(ymm18, ymm23, ymm25); \
    ymm18 = _mm256_broadcast_sd((a01_2 + (p_lda * 7))); \
    ymm26 = _mm256_fmadd_pd(ymm19, ymm23, ymm26); \
    ymm27 = _mm256_fmadd_pd(ymm20, ymm23, ymm27); \
    ymm28 = _mm256_fmadd_pd(ymm21, ymm23, ymm28); \
    ymm29 = _mm256_fmadd_pd(ymm22, ymm23, ymm29); \
    ymm30 = _mm256_fmadd_pd(ymm17, ymm23, ymm30); \
    ymm31 = _mm256_fmadd_pd(ymm18, ymm23, ymm31); \
    \
    a01_2 += 1; \
    b10_2 += cs_b; \
  } \
  /*combine the results of both loops*/ \
  _mm_prefetch((b11 + (0) * cs_b), _MM_HINT_T0); \
  ymm9 = _mm256_add_pd(ymm9, ymm24); \
  _mm_prefetch((b11 + (1) * cs_b), _MM_HINT_T0); \
  ymm10 = _mm256_add_pd(ymm10, ymm25); \
  _mm_prefetch((b11 + (2) * cs_b), _MM_HINT_T0); \
  ymm11 = _mm256_add_pd(ymm11, ymm26); \
  _mm_prefetch((b11 + (3) * cs_b), _MM_HINT_T0); \
  ymm12 = _mm256_add_pd(ymm12, ymm27); \
  _mm_prefetch((b11 + (4) * cs_b), _MM_HINT_T0); \
  ymm13 = _mm256_add_pd(ymm13, ymm28); \
  _mm_prefetch((b11 + (5) * cs_b), _MM_HINT_T0); \
  ymm14 = _mm256_add_pd(ymm14, ymm29); \
  _mm_prefetch((b11 + (6) * cs_b), _MM_HINT_T0); \
  ymm15 = _mm256_add_pd(ymm15, ymm30); \
  _mm_prefetch((b11 + (7) * cs_b), _MM_HINT_T0); \
  ymm16 = _mm256_add_pd(ymm16, ymm31);



#define BLIS_DTRSM_SMALL_GEMM_4nx8m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 8x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); \
    ymm1 = _mm256_loadu_pd((double const *)(b10 + 4)); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/ \
    ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
    ymm10 = _mm256_fmadd_pd(ymm2, ymm1, ymm10); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_4nx4m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 4x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/ \
    ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_4nx3m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 3x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_broadcast_sd((double const *)(b10 + 2)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/ \
    ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_4nx2m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 2x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/ \
    ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_4nx1m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 1x1 block of B10*/ \
    ymm0 = _mm256_broadcast_sd((double const *)b10); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 3)); /*A01[0][3]*/ \
    ymm9 = _mm256_fmadd_pd(ymm2, ymm0, ymm9); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_3nx8m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 8x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); \
    ymm1 = _mm256_loadu_pd((double const *)(b10 + 4)); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    ymm8 = _mm256_fmadd_pd(ymm2, ymm1, ymm8); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_3nx4m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 4x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_3nx3m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 3x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_broadcast_sd((double const *)(b10 + 2)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_3nx2m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 2x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_3nx1m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 1x1 block of B10*/ \
    ymm0 = _mm256_broadcast_sd((double const *)b10); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 2)); /*A01[0][2]*/ \
    ymm7 = _mm256_fmadd_pd(ymm2, ymm0, ymm7); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_2nx8m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 8x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10);     /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    ymm1 = _mm256_loadu_pd((double const *)(b10 + 4)); /*B10[4][0] B10[5][0] B10[6][0] B10[7][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    ymm6 = _mm256_fmadd_pd(ymm2, ymm1, ymm6); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_2nx4m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 4x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_2nx3m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 3x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_broadcast_sd((double const *)(b10 + 2)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_2nx2m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 2x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
   \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
   \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
   \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_2nx1m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 1x1 block of B10*/ \
    ymm0 = _mm256_broadcast_sd((double const *)b10); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 1)); /*A01[0][1]*/ \
    ymm5 = _mm256_fmadd_pd(ymm2, ymm0, ymm5); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_1nx8m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 8x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10);     /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    ymm1 = _mm256_loadu_pd((double const *)(b10 + 4)); /*B10[4][0] B10[5][0] B10[6][0] B10[7][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    ymm4 = _mm256_fmadd_pd(ymm2, ymm1, ymm4); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_1nx4m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 4x1 block of B10*/ \
    ymm0 = _mm256_loadu_pd((double const *)b10); /*B10[0][0] B10[1][0] B10[2][0] B10[3][0]*/ \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_1nx3m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 3x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_broadcast_sd((double const *)(b10 + 2)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_1nx2m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 2x1 block of B10*/ \
    xmm5 = _mm_loadu_pd((double const *)(b10)); \
    ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

#define BLIS_DTRSM_SMALL_GEMM_1nx1m(a01, b10, cs_b, p_lda, k_iter) \
  for (k = 0; k < k_iter; k++) /*loop for number of GEMM operations*/ \
  { \
    /*load 1x1 block of B10*/ \
    ymm0 = _mm256_broadcast_sd((double const *)b10); \
    \
    /*broadcast 1st row of A01*/ \
    ymm2 = _mm256_broadcast_sd((double const *)(a01 + p_lda * 0)); /*A01[0][0]*/ \
    ymm3 = _mm256_fmadd_pd(ymm2, ymm0, ymm3); \
    \
    a01 += 1; /*move to next row*/ \
    b10 += cs_b; \
  }

// endregion - GEMM DTRSM for right variants

// region - pre/post DTRSM macros for right variants

#define BLIS_PRE_DTRSM_SMALL_8x8(AlphaVal, b11, cs_b) \
  /*gemm_output = (B11 * alpha) - gemm_output*/ \
  zmm31 = _mm512_set1_pd(AlphaVal); \
  \
  zmm0  = _mm512_loadu_pd((double const *)(b11 + (0*cs_b))); \
  zmm9  = _mm512_fmsub_pd(zmm0, zmm31, zmm9 ); /*zmm9  = (zmm0 * zmm31) - zmm9*/\
  \
  zmm1  = _mm512_loadu_pd((double const *)(b11 + (1*cs_b))); \
  zmm10 = _mm512_fmsub_pd(zmm1, zmm31, zmm10); /*zmm10 = (zmm1 * zmm31) - zmm10*/\
  \
  zmm2  = _mm512_loadu_pd((double const *)(b11 + (2*cs_b))); \
  zmm11 = _mm512_fmsub_pd(zmm2, zmm31, zmm11); /*zmm11 = (zmm2 * zmm31) - zmm11*/\
  \
  zmm3  = _mm512_loadu_pd((double const *)(b11 + (3*cs_b))); \
  zmm12 = _mm512_fmsub_pd(zmm3, zmm31, zmm12); /*zmm12 = (zmm3 * zmm31) - zmm12*/\
  \
  zmm4  = _mm512_loadu_pd((double const *)(b11 + (4*cs_b))); \
  zmm13 = _mm512_fmsub_pd(zmm4, zmm31, zmm13); /*zmm13 = (zmm4 * zmm31) - zmm13*/\
  \
  zmm5  = _mm512_loadu_pd((double const *)(b11 + (5*cs_b))); \
  zmm14 = _mm512_fmsub_pd(zmm5, zmm31, zmm14); /*zmm14 = (zmm5 * zmm31) - zmm14*/\
  \
  zmm6  = _mm512_loadu_pd((double const *)(b11 + (6*cs_b))); \
  zmm15 = _mm512_fmsub_pd(zmm6, zmm31, zmm15); /*zmm15 = (zmm6 * zmm31) - zmm15*/\
  \
  zmm7  = _mm512_loadu_pd((double const *)(b11 + (7*cs_b))); \
  zmm16 = _mm512_fmsub_pd(zmm7, zmm31, zmm16);

#define BLIS_PRE_DTRSM_SMALL_8x4(AlphaVal, b11, cs_b) \
  /*gemm_output = (B11 * alpha) - gemm_output*/ \
  ymm31 = _mm256_broadcast_sd(&AlphaVal); \
  \
  ymm0  = _mm256_loadu_pd((double const *)(b11 + 0 * cs_b)); \
  ymm9  = _mm256_fmsub_pd(ymm0, ymm31, ymm9); \
  \
  ymm1  = _mm256_loadu_pd((double const *)(b11 + 1 * cs_b)); \
  ymm10 = _mm256_fmsub_pd(ymm1, ymm31, ymm10); \
  \
  ymm2  = _mm256_loadu_pd((double const *)(b11 + 2 * cs_b)); \
  ymm11 = _mm256_fmsub_pd(ymm2, ymm31, ymm11); \
  \
  ymm3  = _mm256_loadu_pd((double const *)(b11 + 3 * cs_b)); \
  ymm12 = _mm256_fmsub_pd(ymm3, ymm31, ymm12); \
  \
  ymm4  = _mm256_loadu_pd((double const *)(b11 + 4 * cs_b)); \
  ymm13 = _mm256_fmsub_pd(ymm4, ymm31, ymm13); \
  \
  ymm5  = _mm256_loadu_pd((double const *)(b11 + 5 * cs_b)); \
  ymm14 = _mm256_fmsub_pd(ymm5, ymm31, ymm14); \
  \
  ymm6  = _mm256_loadu_pd((double const *)(b11 + 6 * cs_b)); \
  ymm15 = _mm256_fmsub_pd(ymm6, ymm31, ymm15); \
  \
  ymm7  = _mm256_loadu_pd((double const *)(b11 + 7 * cs_b)); \
  ymm16 = _mm256_fmsub_pd(ymm7, ymm31, ymm16);

#define BLIS_PRE_DTRSM_SMALL_8x3(AlphaVal, b11, cs_b) \
  /*gemm_output = (B11 * alpha) - gemm_output*/ \
  ymm31 = _mm256_broadcast_sd(&AlphaVal); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 0 * cs_b)); /*xmm5 = {b11[0], b11[1]}*/\
  ymm0 = _mm256_broadcast_sd((b11 + 2 + 0 * cs_b)); /*ymm0 = {b11[2], b11[2], b11[2], b11[2]}*/\
  ymm0 = _mm256_insertf64x2(ymm0, xmm5, 0); /*ymm0 = {b11[0], b11[1], b11[2], b11[2]}*/\
  ymm9 = _mm256_fmsub_pd(ymm0, ymm31, ymm9); /*ymm9 = (ymm0 * ymm31) - ymm9*/\
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 1 * cs_b)); \
  ymm1 = _mm256_broadcast_sd((b11 + 2 + 1 * cs_b)); \
  ymm1 = _mm256_insertf64x2(ymm1, xmm5, 0); \
  ymm10 = _mm256_fmsub_pd(ymm1, ymm31, ymm10); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 2 * cs_b)); \
  ymm2 = _mm256_broadcast_sd((b11 + 2 + 2 * cs_b)); \
  ymm2 = _mm256_insertf64x2(ymm2, xmm5, 0); \
  ymm11 = _mm256_fmsub_pd(ymm2, ymm31, ymm11); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 3 * cs_b)); \
  ymm3 = _mm256_broadcast_sd((b11 + 2 + 3 * cs_b)); \
  ymm3 = _mm256_insertf64x2(ymm3, xmm5, 0); \
  ymm12 = _mm256_fmsub_pd(ymm3, ymm31, ymm12); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 4 * cs_b)); \
  ymm4 = _mm256_broadcast_sd((b11 + 2 + 4 * cs_b)); \
  ymm4 = _mm256_insertf64x2(ymm4, xmm5, 0); \
  ymm13 = _mm256_fmsub_pd(ymm4, ymm31, ymm13); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 5 * cs_b)); \
  ymm5 = _mm256_broadcast_sd((b11 + 2 + 5 * cs_b)); \
  ymm5 = _mm256_insertf64x2(ymm5, xmm5, 0); \
  ymm14 = _mm256_fmsub_pd(ymm5, ymm31, ymm14); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 6 * cs_b)); \
  ymm6 = _mm256_broadcast_sd((b11 + 2 + 6 * cs_b)); \
  ymm6 = _mm256_insertf64x2(ymm6, xmm5, 0); \
  ymm15 = _mm256_fmsub_pd(ymm6, ymm31, ymm15); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 7 * cs_b)); \
  ymm7 = _mm256_broadcast_sd((b11 + 2 + 7 * cs_b)); \
  ymm7 = _mm256_insertf64x2(ymm7, xmm5, 0); \
  ymm16 = _mm256_fmsub_pd(ymm7, ymm31, ymm16);

#define BLIS_PRE_DTRSM_SMALL_8x2(AlphaVal, b11, cs_b) \
  /*gemm_output = (B11 * alpha) - gemm_output*/ \
  ymm31 = _mm256_broadcast_sd(&AlphaVal); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 0 * cs_b)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);\
  ymm9 = _mm256_fmsub_pd(ymm0, ymm31, ymm9); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 1 * cs_b)); \
  ymm1 = _mm256_insertf128_pd(ymm1, xmm5, 0);\
  ymm10 = _mm256_fmsub_pd(ymm1, ymm31, ymm10); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 2 * cs_b)); \
  ymm2 = _mm256_insertf128_pd(ymm2, xmm5, 0);\
  ymm11 = _mm256_fmsub_pd(ymm2, ymm31, ymm11); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 3 * cs_b)); \
  ymm3 = _mm256_insertf128_pd(ymm3, xmm5, 0);\
  ymm12 = _mm256_fmsub_pd(ymm3, ymm31, ymm12); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 4 * cs_b)); \
  ymm4 = _mm256_insertf128_pd(ymm4, xmm5, 0);\
  ymm13 = _mm256_fmsub_pd(ymm4, ymm31, ymm13); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 5 * cs_b)); \
  ymm5 = _mm256_insertf128_pd(ymm5, xmm5, 0);\
  ymm14 = _mm256_fmsub_pd(ymm5, ymm31, ymm14); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 6 * cs_b)); \
  ymm6 = _mm256_insertf128_pd(ymm6, xmm5, 0);\
  ymm15 = _mm256_fmsub_pd(ymm6, ymm31, ymm15); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + 7 * cs_b)); \
  ymm7 = _mm256_insertf128_pd(ymm7, xmm5, 0);\
  ymm16 = _mm256_fmsub_pd(ymm7, ymm31, ymm16);

#define BLIS_PRE_DTRSM_SMALL_8x1(AlphaVal, b11, cs_b) \
  /*gemm_output = (B11 * alpha) - gemm_output*/ \
  ymm31 = _mm256_broadcast_sd(&AlphaVal); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 0 * cs_b)); \
  ymm0 = _mm256_insertf64x2(ymm0, xmm5, 0); \
  ymm9 = _mm256_fmsub_pd(ymm0, ymm31, ymm9); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 1 * cs_b)); \
  ymm1 = _mm256_insertf64x2(ymm1, xmm5, 0); \
  ymm10 = _mm256_fmsub_pd(ymm1, ymm31, ymm10); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 2 * cs_b)); \
  ymm2 = _mm256_insertf64x2(ymm2, xmm5, 0); \
  ymm11 = _mm256_fmsub_pd(ymm2, ymm31, ymm11); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 3 * cs_b)); \
  ymm3 = _mm256_insertf64x2(ymm3, xmm5, 0); \
  ymm12 = _mm256_fmsub_pd(ymm3, ymm31, ymm12); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 4 * cs_b)); \
  ymm4 = _mm256_insertf64x2(ymm4, xmm5, 0); \
  ymm13 = _mm256_fmsub_pd(ymm4, ymm31, ymm13); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 5 * cs_b)); \
  ymm5 = _mm256_insertf64x2(ymm5, xmm5, 0); \
  ymm14 = _mm256_fmsub_pd(ymm5, ymm31, ymm14); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 6 * cs_b)); \
  ymm6 = _mm256_insertf64x2(ymm6, xmm5, 0); \
  ymm15 = _mm256_fmsub_pd(ymm6, ymm31, ymm15); \
  \
  xmm5 = _mm_loadl_pd(xmm5, (double const *)(b11 + 7 * cs_b)); \
  ymm7 = _mm256_insertf64x2(ymm7, xmm5, 0); \
  ymm16 = _mm256_fmsub_pd(ymm7, ymm31, ymm16);

#define BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal)); \
  \
  ymm0 = _mm256_loadu_pd((double const *)b11); \
  ymm1 = _mm256_loadu_pd((double const *)(b11 + 4)); \
  \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4); \
  \
  ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b)); \
  ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4)); \
  \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5); \
  ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6); \
  \
  ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b * 2)); \
  ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b * 2 + 4)); \
  \
  ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7); \
  ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8); \
  \
  ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b * 3)); \
  ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b * 3 + 4)); \
  \
  ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9); \
  ymm10 = _mm256_fmsub_pd(ymm1, ymm15, ymm10);

#define BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b * 2)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b * 2 + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_3M(b11, cs_b) \
 \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5); \
  _mm_storel_pd((b11 + 2), _mm256_extractf128_pd(ymm3, 1)); \
  xmm5 = _mm256_castpd256_pd128(ymm5); \
  _mm_storeu_pd((double *)(b11 + cs_b), xmm5); \
  _mm_storel_pd((b11 + cs_b + 2), _mm256_extractf128_pd(ymm5, 1)); \
  xmm5 = _mm256_castpd256_pd128(ymm7); \
  _mm_storeu_pd((double *)(b11 + cs_b * 2), xmm5); \
  _mm_storel_pd((b11 + cs_b * 2 + 2), _mm256_extractf128_pd(ymm7, 1));

#define BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b * 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_2M(b11, cs_b) \
 \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5); \
  xmm5 = _mm256_castpd256_pd128(ymm5); \
  _mm_storeu_pd((double *)(b11 + cs_b), xmm5); \
  xmm5 = _mm256_castpd256_pd128(ymm7); \
  _mm_storeu_pd((double *)(b11 + cs_b * 2), xmm5);

#define BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  ymm0 = _mm256_broadcast_sd((double const *)b11); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b)); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5); \
  \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b * 2)); \
  ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);

#define BLIS_POST_DTRSM_SMALL_3N_1M(b11, cs_b) \
 \
  _mm_storel_pd((b11 + cs_b * 0), _mm256_castpd256_pd128(ymm3)); \
  _mm_storel_pd((b11 + cs_b * 1), _mm256_castpd256_pd128(ymm5)); \
  _mm_storel_pd((b11 + cs_b * 2), _mm256_castpd256_pd128(ymm7));

#define BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b * 1)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b * 1 + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_3M(b11, cs_b) \
 \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5); \
  _mm_storel_pd((b11 + 2), _mm256_extractf128_pd(ymm3, 1)); \
  xmm5 = _mm256_castpd256_pd128(ymm5); \
  _mm_storeu_pd((double *)(b11 + cs_b * 1), xmm5); \
  _mm_storel_pd((b11 + cs_b * 1 + 2), _mm256_extractf128_pd(ymm5, 1));

#define BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11 + cs_b * 1)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_2M(b11, cs_b) \
 \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5); \
  xmm5 = _mm256_castpd256_pd128(ymm5); \
  _mm_storeu_pd((double *)(b11 + cs_b * 1), xmm5);

#define BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  ymm0 = _mm256_broadcast_sd((double const *)b11); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3); \
  \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b)); \
  ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);

#define BLIS_POST_DTRSM_SMALL_2N_1M(b11, cs_b) \
 \
  _mm_storel_pd(b11, _mm256_castpd256_pd128(ymm3)); \
  _mm_storel_pd((b11 + cs_b * 1), _mm256_castpd256_pd128(ymm5));

#define BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_broadcast_sd((double const *)(b11 + 2)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_3M(b11, cs_b) \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5); \
  _mm_storel_pd((b11 + 2), _mm256_extractf128_pd(ymm3, 1));

#define BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  xmm5 = _mm_loadu_pd((double const *)(b11)); \
  ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_2M(b11, cs_b) \
 \
  xmm5 = _mm256_castpd256_pd128(ymm3); \
  _mm_storeu_pd((double *)(b11), xmm5);

#define BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal, b11, cs_b) \
  ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); /*register to hold alpha*/ \
  \
  ymm0 = _mm256_broadcast_sd((double const *)b11); \
  ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);

#define BLIS_POST_DTRSM_SMALL_1N_1M(b11, cs_b) \
 \
  _mm_storel_pd(b11, _mm256_castpd256_pd128(ymm3));

// endregion - pre/post DTRSM macros for right variants

// RUNN - RLTN
BLIS_INLINE err_t bli_dtrsm_small_XAltB_XAuB_AVX512
     (
       obj_t*   AlphaObj,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     )
{
  dim_t m = bli_obj_length(b); //number of rows
  dim_t n = bli_obj_width(b); // number of columns
  dim_t d_mr = 8, d_nr = 8;

  bool transa = bli_obj_has_trans(a);
  dim_t cs_a, rs_a;
  double ones = 1.0;

  // Swap rs_a & cs_a in case of non-transpose.
  if (transa)
  {
    cs_a = bli_obj_col_stride(a); // column stride of A
    rs_a = bli_obj_row_stride(a); // row stride of A
  }
  else
  {
    cs_a = bli_obj_row_stride(a); // row stride of A
    rs_a = bli_obj_col_stride(a); // column stride of A
  }

  dim_t cs_b = bli_obj_col_stride(b); // column stride of B

  dim_t i, j, k;
  dim_t k_iter;

  bool is_unitdiag = bli_obj_has_unit_diag(a);

  double AlphaVal = *(double *)AlphaObj->buffer;
  double *restrict L = a->buffer; // pointer to matrix A
  double *B = bli_obj_buffer_at_off(b); // pointer to matrix B

  double *a01, *a11, *b10, *b11; // pointers for GEMM and TRSM blocks

  gint_t required_packing_A = 1;
  mem_t local_mem_buf_A_s = {0};
  double *D_A_pack = NULL; // pointer to A01 pack buffer
  double d11_pack[d_mr] __attribute__((aligned(64))); // buffer for diagonal A pack
  rntm_t rntm;

  bli_rntm_init_from_global(&rntm);
  bli_rntm_set_num_threads_only(1, &rntm);
  bli_membrk_rntm_set_membrk(&rntm);

  siz_t buffer_size = bli_pool_block_size(
    bli_membrk_pool(
      bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
      bli_rntm_membrk(&rntm)));

  if ((d_nr * n * sizeof(double)) > buffer_size)
    return BLIS_NOT_YET_IMPLEMENTED;

  if (required_packing_A == 1)
  {
    // Get the buffer from the pool.
    bli_membrk_acquire_m(&rntm,
               buffer_size,
               BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
               &local_mem_buf_A_s); // acquire memory for A01 pack
    if (FALSE == bli_mem_is_alloc(&local_mem_buf_A_s))
      return BLIS_NULL_POINTER;
    D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
    if (NULL == D_A_pack)
      return BLIS_NULL_POINTER;
  }

  __m512d zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11;
  __m512d zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
  __m512d zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
  __m256d ymm12, ymm13, ymm14, ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21;
  __m256d ymm22, ymm23, ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31;
  __m128d xmm5, xmm0;

  /*
    Performs solving TRSM for 8 rows at a time from  0 to n/8 in steps of d_nr
    a. Load and pack A (a01 block), the size of packing 8x8 to 8x(n-8)
        First there will be no GEMM and no packing of a01 because it is only TRSM
    b. Using packed a01 block and b10 block perform GEMM operation
    c. Use GEMM outputs, perform TRSM operation using a11, b11 and update B
    d. Repeat b for m cols of B in steps of d_mr
  */
  for (j = 0; (j + d_nr - 1) < n; j += d_nr) //loop along 'N' direction
  {
    a01 = L + j * rs_a;            //pointer to block of A to be used in GEMM
    a11 = L + j * cs_a + j * rs_a; //pointer to block of A to be used for TRSM

    dim_t p_lda = j;               //packed leading dimension

    // perform copy of A to packed buffer D_A_pack
    if (transa)
    {
      /*
      Pack current A block (a01) into packed buffer memory D_A_pack
        a. This a10 block is used in GEMM portion only and this
            a01 block size will be increasing by d_nr for every next iteration
            until it reaches 8x(n-8) which is the maximum GEMM alone block size in A
        b. This packed buffer is reused to calculate all m cols of B matrix
      */
      bli_dtrsm_small_pack_avx512('R', j, 1, a01, cs_a, D_A_pack, p_lda, d_nr);
      /*
        Pack 8 diagonal elements of A block into an array
        a. This helps to utilize cache line efficiently in TRSM operation
        b. store ones when input is unit diagonal
      */
      dtrsm_small_pack_diag_element_avx512(is_unitdiag, a11, cs_a, d11_pack, d_nr);
    }
    else
    {
      bli_dtrsm_small_pack_avx512('R', j, 0, a01, rs_a, D_A_pack, p_lda, d_nr);
      dtrsm_small_pack_diag_element_avx512(is_unitdiag, a11, rs_a, d11_pack, d_nr);
    }

    /*
      a. Perform GEMM using a01, b10.
      b. Perform TRSM on a11, b11
      c. This loop GEMM+TRSM loops operates with 8x6 block size
          along m dimension for every d_mr columns of B10 where
          packed A buffer is reused in computing all m cols of B.
      d. Same approach is used in remaining fringe cases.
    */
    for (i = 0; (i + d_mr - 1) < m; i += d_mr) //loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a;   //pointer to block of A to be used for TRSM
      b10 = B + i;                     //pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;          //pointer to block of B to be used for TRSM

      k_iter = j;
      BLIS_SET_ZMM_REG_ZEROS
      /*
        Perform GEMM between a01 and b10 blocks
        For first iteration there will be no GEMM operation
        where k_iter are zero
      */
      BLIS_DTRSM_SMALL_GEMM_8nx8m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11);
      /*
        Load b11 of size 8x8 and multiply with alpha
        Add the GEMM output to b11
        and perform TRSM operation.
      */
      BLIS_PRE_DTRSM_SMALL_8x8(AlphaVal, b11, cs_b)



      /*
        Compute 8x8 TRSM block by using GEMM block output in register
        a. The 8x8 input (gemm outputs) are stored in combinations of zmm registers
            row      :   0     1    2      3     4     5     6     7
            register : zmm9  zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16
        b. Towards the end TRSM output will be stored back into b11
      */

      /*
      *                                        to n-1
      *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[i][N]  } /A11[N][N]
      *                                       from i=0
      *
      *  For example 5th column (B11[5]) -= ((B11[0] * A11[0][5]) + (B11[1] * A11[2][5]) +
      *                                      (B11[2] * A11[2][5]) + (B11[3] * A11[3][5]) +
      *                                      (B11[4] * A11[4][5])) / A11[5][5]
      *                          zmm14   -= ((zmm9   * A11[0][5]) + (zmm10  * A11[2][5]) +
      *                                      (zmm11  * A11[2][5]) + (zmm12  * A11[3][5]) +
      *                                      (zmm13  * A11[4][5])) / A11[5][5]
      */


      // extract a00
      zmm0 = _mm512_set1_pd(*(d11_pack + 0));
      zmm9 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm9, zmm0); // zmm9 /= zmm0
      _mm512_storeu_pd((double *)(b11 + (0 * cs_b)), zmm9);

      // extract a11
      zmm1 = _mm512_set1_pd(*(d11_pack + 1));
      zmm2 = _mm512_set1_pd(*(a11 + (1 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm2, zmm9, zmm10);
      zmm3 = _mm512_set1_pd(*(a11 + (2 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm3, zmm9, zmm11);
      zmm4 = _mm512_set1_pd(*(a11 + (3 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm4, zmm9, zmm12);
      zmm5 = _mm512_set1_pd(*(a11 + (4 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm5, zmm9, zmm13);
      zmm6 = _mm512_set1_pd(*(a11 + (5 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm6, zmm9, zmm14); // zmm14 -= A11[0][5] * zmm9
      zmm7 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm7, zmm9, zmm15);
      zmm8 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm8, zmm9, zmm16);
      zmm10 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm10, zmm1);
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (1 * cs_b)), zmm10);

      // extract a22
      zmm0 = _mm512_set1_pd(*(d11_pack + 2));
      zmm2 = _mm512_set1_pd(*(a11 + (2 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm2, zmm10, zmm11);
      zmm3 = _mm512_set1_pd(*(a11 + (3 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm3, zmm10, zmm12);
      zmm4 = _mm512_set1_pd(*(a11 + (4 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm4, zmm10, zmm13);
      zmm5 = _mm512_set1_pd(*(a11 + (5 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm5, zmm10, zmm14); // zmm14 -= A11[1][5] * zmm10
      zmm6 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm6, zmm10, zmm15);
      zmm7 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm7, zmm10, zmm16);
      zmm11 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm11, zmm0);
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (2 * cs_b)), zmm11);

      // extract a33
      zmm1 = _mm512_set1_pd(*(d11_pack + 3));
      zmm2 = _mm512_set1_pd(*(a11 + (3 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm2, zmm11, zmm12);
      zmm3 = _mm512_set1_pd(*(a11 + (4 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm3, zmm11, zmm13);
      zmm4 = _mm512_set1_pd(*(a11 + (5 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm4, zmm11, zmm14); // zmm14 -= A11[2][5] * zmm11
      zmm5 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm5, zmm11, zmm15);
      zmm6 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm6, zmm11, zmm16);
      zmm12 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm12, zmm1);
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (3 * cs_b)), zmm12);

      // extract a44
      zmm0 = _mm512_set1_pd(*(d11_pack + 4));
      zmm2 = _mm512_set1_pd(*(a11 + (4 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm2, zmm12, zmm13);
      zmm3 = _mm512_set1_pd(*(a11 + (5 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm3, zmm12, zmm14); // zmm14 -= A11[3][5] * zmm12
      zmm4 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm4, zmm12, zmm15);
      zmm5 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm5, zmm12, zmm16);
      zmm13 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm13, zmm0);
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (4 * cs_b)), zmm13);

      // extract a55
      zmm1 = _mm512_set1_pd(*(d11_pack + 5));
      zmm2 = _mm512_set1_pd(*(a11 + (5 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm2, zmm13, zmm14); // zmm14 -= A11[4][5] * zmm13
      zmm3 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm3, zmm13, zmm15);
      zmm4 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm4, zmm13, zmm16);
      zmm14 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm14, zmm1); // zmm14 /= A11[5][5]
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (5 * cs_b)), zmm14);

      // extract a66
      zmm0 = _mm512_set1_pd(*(d11_pack + 6));
      zmm2 = _mm512_set1_pd(*(a11 + (6 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm2, zmm14, zmm15);
      zmm3 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm3, zmm14, zmm16);
      zmm15 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm15, zmm0);
      a11 += cs_a;
      _mm512_storeu_pd((double *)(b11 + (6 * cs_b)), zmm15);

      // extract a77
      zmm1 = _mm512_set1_pd(*(d11_pack + 7));
      zmm2 = _mm512_set1_pd(*(a11 + (7 * rs_a)));
      zmm16 = _mm512_fnmadd_pd(zmm2, zmm15, zmm16);
      zmm16 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm16, zmm1);
      _mm512_storeu_pd((double *)(b11 + (7 * cs_b)), zmm16);
    }
    dim_t m_remainder = m - i;
    if(m_remainder)
    {
      if (m_remainder >= 4) //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + j * cs_a + j * rs_a;   //pointer to block of A to be used for TRSM
        b10 = B + i;                     //pointer to block of B to be used in GEMM
        b11 = B + i + j * cs_b;          //pointer to block of B to be used for TRSM

        k_iter = j;
        BLIS_SET_YMM_REG_ZEROS_AVX512
        /*
          Perform GEMM between a01 and b10 blocks
          For first iteration there will be no GEMM operation
          where k_iter are zero
        */
        BLIS_DTRSM_SMALL_GEMM_8nx4m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        /*
          Load b11 of size 8x4 and multiply with alpha
          Add the GEMM output to b11
          and perform TRSM operation.
        */
        BLIS_PRE_DTRSM_SMALL_8x4(AlphaVal, b11, cs_b)

        /*
          Compute 8x4 TRSM block by using GEMM block output in register
          a. The 8x4 input (gemm outputs) are stored in combinations of ymm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                        to n-1
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[i][N]  } /A11[N][N]
        *                                       from i=0
        *
        *  For example 5th column (B11[5]) -= ((B11[0] * A11[0][5]) + (B11[1] * A11[2][5]) +
        *                                      (B11[2] * A11[2][5]) + (B11[3] * A11[3][5]) +
        *                                      (B11[4] * A11[4][5])) / A11[5][5]
        */

        // extract a00
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm256_storeu_pd((double *)(b11 + (0 * cs_b)),  ymm9);

        // extract a11
        ymm1 = _mm256_broadcast_sd((d11_pack + 1));
        ymm2 = _mm256_broadcast_sd((a11 + (1 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
        ymm3 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm3, ymm9, ymm11);
        ymm4 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm4, ymm9, ymm12);
        ymm5 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm5, ymm9, ymm13);
        ymm6 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm6, ymm9, ymm14);
        ymm7 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm7, ymm9, ymm15);
        ymm8 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm8, ymm9, ymm16);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (1 * cs_b)), ymm10);

        // extract a22
        ymm0 = _mm256_broadcast_sd((d11_pack + 2));
        ymm2 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
        ymm3 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm3, ymm10, ymm12);
        ymm4 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm4, ymm10, ymm13);
        ymm5 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm5, ymm10, ymm14);
        ymm6 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm6, ymm10, ymm15);
        ymm7 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm7, ymm10, ymm16);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (2 * cs_b)), ymm11);

        // extract a33
        ymm1 = _mm256_broadcast_sd((d11_pack + 3));
        ymm2 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm2, ymm11, ymm12);
        ymm3 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm3, ymm11, ymm13);
        ymm4 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm4, ymm11, ymm14);
        ymm5 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm5, ymm11, ymm15);
        ymm6 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm6, ymm11, ymm16);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (3 * cs_b)), ymm12);

        // extract a44
        ymm0 = _mm256_broadcast_sd((d11_pack + 4));
        ymm2 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm2, ymm12, ymm13);
        ymm3 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm3, ymm12, ymm14);
        ymm4 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm4, ymm12, ymm15);
        ymm5 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm5, ymm12, ymm16);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (4 * cs_b)), ymm13);

        // extract a55
        ymm1 = _mm256_broadcast_sd((d11_pack + 5));
        ymm2 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm2, ymm13, ymm14);
        ymm3 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm3, ymm13, ymm15);
        ymm4 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm4, ymm13, ymm16);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (5 * cs_b)), ymm14);

        // extract a66
        ymm0 = _mm256_broadcast_sd((d11_pack + 6));
        ymm2 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm2, ymm14, ymm15);
        ymm3 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm3, ymm14, ymm16);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm0);
        a11 += cs_a;
        _mm256_storeu_pd((double *)(b11 + (6 * cs_b)), ymm15);

        // extract a77
        ymm1 = _mm256_broadcast_sd((d11_pack + 7));
        ymm2 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm2, ymm15, ymm16);
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm1);
        _mm256_storeu_pd((double *)(b11 + (7 * cs_b)), ymm16);
        m_remainder -= 4;
        i += 4;
      }
      if (m_remainder == 3) //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + j * cs_a + j * rs_a;   //pointer to block of A to be used for TRSM
        b10 = B + i;                     //pointer to block of B to be used in GEMM
        b11 = B + i + j * cs_b;          //pointer to block of B to be used for TRSM

        k_iter = j;
        BLIS_SET_YMM_REG_ZEROS_AVX512
        /*
          Perform GEMM between a01 and b10 blocks
          For first iteration there will be no GEMM operation
          where k_iter are zero
        */
        BLIS_DTRSM_SMALL_GEMM_8nx3m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        /*
          Load b11 of size 8x3 and multiply with alpha
          Add the GEMM output to b11
          and perform TRSM operation.
        */
        BLIS_PRE_DTRSM_SMALL_8x3(AlphaVal, b11, cs_b)
        /*
          Compute 8x3 TRSM block by using GEMM block output in register
          a. The 8x3 input (gemm outputs) are stored in combinations of ymm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                        to n-1
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[i][N]  } /A11[N][N]
        *                                       from i=0
        *
        *  For example 5th column (B11[5]) -= ((B11[0] * A11[0][5]) + (B11[1] * A11[2][5]) +
        *                                      (B11[2] * A11[2][5]) + (B11[3] * A11[3][5]) +
        *                                      (B11[4] * A11[4][5])) / A11[5][5]
        */


        // extract a00
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm_storeu_pd((double *)(b11 + (0 * cs_b)), _mm256_castpd256_pd128(ymm9));
        _mm_storel_pd((double *)(b11 + (0 * cs_b) + 2), _mm256_extractf64x2_pd(ymm9, 1));

        // extract a11
        ymm1 = _mm256_broadcast_sd((d11_pack + 1));
        ymm2 = _mm256_broadcast_sd((a11 + (1 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
        ymm3 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm3, ymm9, ymm11);
        ymm4 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm4, ymm9, ymm12);
        ymm5 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm5, ymm9, ymm13);
        ymm6 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm6, ymm9, ymm14);
        ymm7 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm7, ymm9, ymm15);
        ymm8 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm8, ymm9, ymm16);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (1 * cs_b)), _mm256_castpd256_pd128(ymm10));
        _mm_storel_pd((double *)(b11 + (1 * cs_b) + 2), _mm256_extractf64x2_pd(ymm10, 1));

        // extract a22
        ymm0 = _mm256_broadcast_sd((d11_pack + 2));
        ymm2 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
        ymm3 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm3, ymm10, ymm12);
        ymm4 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm4, ymm10, ymm13);
        ymm5 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm5, ymm10, ymm14);
        ymm6 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm6, ymm10, ymm15);
        ymm7 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm7, ymm10, ymm16);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (2 * cs_b)), _mm256_castpd256_pd128(ymm11));
        _mm_storel_pd((double *)(b11 + (2 * cs_b) + 2), _mm256_extractf64x2_pd(ymm11, 1));

        // extract a33
        ymm1 = _mm256_broadcast_sd((d11_pack + 3));
        ymm2 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm2, ymm11, ymm12);
        ymm3 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm3, ymm11, ymm13);
        ymm4 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm4, ymm11, ymm14);
        ymm5 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm5, ymm11, ymm15);
        ymm6 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm6, ymm11, ymm16);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (3 * cs_b)), _mm256_castpd256_pd128(ymm12));
        _mm_storel_pd((double *)(b11 + (3 * cs_b) + 2), _mm256_extractf64x2_pd(ymm12, 1));

        // extract a44
        ymm0 = _mm256_broadcast_sd((d11_pack + 4));
        ymm2 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm2, ymm12, ymm13);
        ymm3 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm3, ymm12, ymm14);
        ymm4 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm4, ymm12, ymm15);
        ymm5 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm5, ymm12, ymm16);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (4 * cs_b)), _mm256_castpd256_pd128(ymm13));
        _mm_storel_pd((double *)(b11 + (4 * cs_b) + 2), _mm256_extractf64x2_pd(ymm13, 1));

        // extract a55
        ymm1 = _mm256_broadcast_sd((d11_pack + 5));
        ymm2 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm2, ymm13, ymm14);
        ymm3 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm3, ymm13, ymm15);
        ymm4 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm4, ymm13, ymm16);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (5 * cs_b)), _mm256_castpd256_pd128(ymm14));
        _mm_storel_pd((double *)(b11 + (5 * cs_b) + 2), _mm256_extractf64x2_pd(ymm14, 1));

        // extract a66
        ymm0 = _mm256_broadcast_sd((d11_pack + 6));
        ymm2 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm2, ymm14, ymm15);
        ymm3 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm3, ymm14, ymm16);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (6 * cs_b)), _mm256_castpd256_pd128(ymm15));
        _mm_storel_pd((double *)(b11 + (6 * cs_b) + 2), _mm256_extractf64x2_pd(ymm15, 1));

        // extract a77
        ymm1 = _mm256_broadcast_sd((d11_pack + 7));
        ymm2 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm2, ymm15, ymm16);
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm1);
        _mm_storeu_pd((double *)(b11 + (7 * cs_b)), _mm256_castpd256_pd128(ymm16));
        _mm_storel_pd((double *)(b11 + (7 * cs_b) + 2), _mm256_extractf64x2_pd(ymm16, 1));
        m_remainder -= 3;
        i += 3;
      }
      else if (m_remainder == 2)
      {
        a01 = D_A_pack;
        a11 = L + j * cs_a + j * rs_a;
        b10 = B + i;
        b11 = B + i + j * cs_b;

        k_iter = j;
        BLIS_SET_YMM_REG_ZEROS_AVX512
        BLIS_DTRSM_SMALL_GEMM_8nx2m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        BLIS_PRE_DTRSM_SMALL_8x2(AlphaVal, b11, cs_b)
        /*
          Compute 8x2 TRSM block by using GEMM block output in register
          a. The 8x2 input (gemm outputs) are stored in combinations of zmm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                        to n-1
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[i][N]  } /A11[N][N]
        *                                       from i=0
        *
        *  For example 5th column (B11[5]) -= ((B11[0] * A11[0][5]) + (B11[1] * A11[2][5]) +
        *                                      (B11[2] * A11[2][5]) + (B11[3] * A11[3][5]) +
        *                                      (B11[4] * A11[4][5])) / A11[5][5]
        */
        // extract a00
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm_storeu_pd((double *)(b11 + (0 * cs_b)), _mm256_castpd256_pd128(ymm9));

        // extract a11
        ymm1 = _mm256_broadcast_sd((d11_pack + 1));
        ymm2 = _mm256_broadcast_sd((a11 + (1 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
        ymm3 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm3, ymm9, ymm11);
        ymm4 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm4, ymm9, ymm12);
        ymm5 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm5, ymm9, ymm13);
        ymm6 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm6, ymm9, ymm14);
        ymm7 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm7, ymm9, ymm15);
        ymm8 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm8, ymm9, ymm16);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (1 * cs_b)), _mm256_castpd256_pd128(ymm10));

        // extract a22
        ymm0 = _mm256_broadcast_sd((d11_pack + 2));
        ymm2 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
        ymm3 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm3, ymm10, ymm12);
        ymm4 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm4, ymm10, ymm13);
        ymm5 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm5, ymm10, ymm14);
        ymm6 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm6, ymm10, ymm15);
        ymm7 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm7, ymm10, ymm16);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (2 * cs_b)), _mm256_castpd256_pd128(ymm11));

        // extract a33
        ymm1 = _mm256_broadcast_sd((d11_pack + 3));
        ymm2 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm2, ymm11, ymm12);
        ymm3 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm3, ymm11, ymm13);
        ymm4 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm4, ymm11, ymm14);
        ymm5 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm5, ymm11, ymm15);
        ymm6 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm6, ymm11, ymm16);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (3 * cs_b)), _mm256_castpd256_pd128(ymm12));

        // extract a44
        ymm0 = _mm256_broadcast_sd((d11_pack + 4));
        ymm2 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm2, ymm12, ymm13);
        ymm3 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm3, ymm12, ymm14);
        ymm4 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm4, ymm12, ymm15);
        ymm5 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm5, ymm12, ymm16);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (4 * cs_b)), _mm256_castpd256_pd128(ymm13));

        // extract a55
        ymm1 = _mm256_broadcast_sd((d11_pack + 5));
        ymm2 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm2, ymm13, ymm14);
        ymm3 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm3, ymm13, ymm15);
        ymm4 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm4, ymm13, ymm16);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (5 * cs_b)), _mm256_castpd256_pd128(ymm14));

        // extract a66
        ymm0 = _mm256_broadcast_sd((d11_pack + 6));
        ymm2 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm2, ymm14, ymm15);
        ymm3 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm3, ymm14, ymm16);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm0);
        a11 += cs_a;
        _mm_storeu_pd((double *)(b11 + (6 * cs_b)), _mm256_castpd256_pd128(ymm15));

        // extract a77
        ymm1 = _mm256_broadcast_sd((d11_pack + 7));
        ymm2 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
        ymm16 = _mm256_fnmadd_pd(ymm2, ymm15, ymm16);
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm1);
        _mm_storeu_pd((double *)(b11 + (7 * cs_b)), _mm256_castpd256_pd128(ymm16));
        m_remainder -= 2;
        i += 2;
      }
      else if (m_remainder == 1)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a;
      b10 = B + i;
      b11 = B + i + j * cs_b;

      k_iter = j;
      BLIS_SET_YMM_REG_ZEROS_AVX512
      BLIS_DTRSM_SMALL_GEMM_8nx1m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
      BLIS_PRE_DTRSM_SMALL_8x1(AlphaVal, b11, cs_b)
      /*
        Compute 8x1 TRSM block by using GEMM block output in register
        a. The 8x1 input (gemm outputs) are stored in combinations of zmm registers
            row      :   0     1    2      3     4     5     6     7
            register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
        b. Towards the end TRSM output will be stored back into b11
      */

      /*
      *                                        to n-1
      *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[i][N]  } /A11[N][N]
      *                                       from i=0
      *
      *  For example 5th column (B11[5]) -= ((B11[0] * A11[0][5]) + (B11[1] * A11[2][5]) +
      *                                      (B11[2] * A11[2][5]) + (B11[3] * A11[3][5]) +
      *                                      (B11[4] * A11[4][5])) / A11[5][5]
      */
      // extract a00
      ymm0 = _mm256_broadcast_sd((d11_pack + 0));
      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
      _mm_storel_pd((double *)(b11 + (0 * cs_b)), _mm256_castpd256_pd128(ymm9));

      // extract a11
      ymm1 = _mm256_broadcast_sd((d11_pack + 1));
      ymm2 = _mm256_broadcast_sd((a11 + (1 * rs_a)));
      ymm10 = _mm256_fnmadd_pd(ymm2, ymm9, ymm10);
      ymm3 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
      ymm11 = _mm256_fnmadd_pd(ymm3, ymm9, ymm11);
      ymm4 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
      ymm12 = _mm256_fnmadd_pd(ymm4, ymm9, ymm12);
      ymm5 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
      ymm13 = _mm256_fnmadd_pd(ymm5, ymm9, ymm13);
      ymm6 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
      ymm14 = _mm256_fnmadd_pd(ymm6, ymm9, ymm14);
      ymm7 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm7, ymm9, ymm15);
      ymm8 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm8, ymm9, ymm16);
      ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (1 * cs_b)), _mm256_castpd256_pd128(ymm10));

      // extract a22
      ymm0 = _mm256_broadcast_sd((d11_pack + 2));
      ymm2 = _mm256_broadcast_sd((a11 + (2 * rs_a)));
      ymm11 = _mm256_fnmadd_pd(ymm2, ymm10, ymm11);
      ymm3 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
      ymm12 = _mm256_fnmadd_pd(ymm3, ymm10, ymm12);
      ymm4 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
      ymm13 = _mm256_fnmadd_pd(ymm4, ymm10, ymm13);
      ymm5 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
      ymm14 = _mm256_fnmadd_pd(ymm5, ymm10, ymm14);
      ymm6 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm6, ymm10, ymm15);
      ymm7 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm7, ymm10, ymm16);
      ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm0);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (2 * cs_b)), _mm256_castpd256_pd128(ymm11));

      // extract a33
      ymm1 = _mm256_broadcast_sd((d11_pack + 3));
      ymm2 = _mm256_broadcast_sd((a11 + (3 * rs_a)));
      ymm12 = _mm256_fnmadd_pd(ymm2, ymm11, ymm12);
      ymm3 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
      ymm13 = _mm256_fnmadd_pd(ymm3, ymm11, ymm13);
      ymm4 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
      ymm14 = _mm256_fnmadd_pd(ymm4, ymm11, ymm14);
      ymm5 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm5, ymm11, ymm15);
      ymm6 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm6, ymm11, ymm16);
      ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (3 * cs_b)), _mm256_castpd256_pd128(ymm12));

      // extract a44
      ymm0 = _mm256_broadcast_sd((d11_pack + 4));
      ymm2 = _mm256_broadcast_sd((a11 + (4 * rs_a)));
      ymm13 = _mm256_fnmadd_pd(ymm2, ymm12, ymm13);
      ymm3 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
      ymm14 = _mm256_fnmadd_pd(ymm3, ymm12, ymm14);
      ymm4 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm4, ymm12, ymm15);
      ymm5 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm5, ymm12, ymm16);
      ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm0);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (4 * cs_b)), _mm256_castpd256_pd128(ymm13));

      // extract a55
      ymm1 = _mm256_broadcast_sd((d11_pack + 5));
      ymm2 = _mm256_broadcast_sd((a11 + (5 * rs_a)));
      ymm14 = _mm256_fnmadd_pd(ymm2, ymm13, ymm14);
      ymm3 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm3, ymm13, ymm15);
      ymm4 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm4, ymm13, ymm16);
      ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (5 * cs_b)), _mm256_castpd256_pd128(ymm14));

      // extract a66
      ymm0 = _mm256_broadcast_sd((d11_pack + 6));
      ymm2 = _mm256_broadcast_sd((a11 + (6 * rs_a)));
      ymm15 = _mm256_fnmadd_pd(ymm2, ymm14, ymm15);
      ymm3 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm3, ymm14, ymm16);
      ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm0);
      a11 += cs_a;
      _mm_storel_pd((double *)(b11 + (6 * cs_b)), _mm256_castpd256_pd128(ymm15));

      // extract a77
      ymm1 = _mm256_broadcast_sd((d11_pack + 7));
      ymm2 = _mm256_broadcast_sd((a11 + (7 * rs_a)));
      ymm16 = _mm256_fnmadd_pd(ymm2, ymm15, ymm16);
      ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm1);
      _mm_storel_pd((double *)(b11 + (7 * cs_b)), _mm256_castpd256_pd128(ymm16));
      m_remainder -= 1;
      i += 1;
    }
    }
  }

  dim_t n_remainder = n - j;

  if (n_remainder >= 4)
  {
    a01 = L + j * rs_a;      // pointer to block of A to be used in GEMM
    a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = j; // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + (p_lda * 0)));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + (p_lda * 1)));
        bli_dcopys(*(a01 + rs_a * 2), *(ptr_a10_dup + (p_lda * 2)));
        bli_dcopys(*(a01 + rs_a * 3), *(ptr_a10_dup + (p_lda * 3)));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = p_lda / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 2) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 3) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 3) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 2) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 3) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 3) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 2) + 2));
        ymm3 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 3) + 3));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 2) + 2));
        ymm3 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 3) + 3));
      }

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = 0; (i + d_mr - 1) < m; i += d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx8m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));

      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);
      ymm8 = _mm256_fnmadd_pd(ymm1, ymm4, ymm8);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));

      ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);
      ymm10 = _mm256_fnmadd_pd(ymm1, ymm4, ymm10);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));

      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);
      ymm8 = _mm256_fnmadd_pd(ymm1, ymm6, ymm8);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));

      ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);
      ymm10 = _mm256_fnmadd_pd(ymm1, ymm6, ymm10);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
      ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

      a11 += cs_a;

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));

      ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);
      ymm10 = _mm256_fnmadd_pd(ymm1, ymm8, ymm10);

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
      ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b + 4)), ymm6);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2) + 4), ymm8);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3)), ymm9);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3) + 4), ymm10);
    }

    dim_t m_remainder = m - i;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal)); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 3)));
      // B11[0][3] B11[1][3] B11[2][3] B11[3][3]
      ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
      // B11[0-3][3] * alpha -= ymm6

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      a11 += cs_a;

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3)), ymm9);

      m_remainder -= 4;
      i += 4;
    }

    if (m_remainder == 3)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx3m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal)); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      xmm5 = _mm_loadu_pd((double const *)(b11 + (cs_b * 3)));
      ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 3) + 2));
      ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
      ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
      // B11[0-3][3] * alpha -= ymm6

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      a11 += cs_a;

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

      _mm_storeu_pd((double *)b11, _mm256_castpd256_pd128(ymm3));
      _mm_storeu_pd((double *)(b11 + cs_b), _mm256_castpd256_pd128(ymm5));
      _mm_storeu_pd((double *)(b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
      _mm_storeu_pd((double *)(b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

      _mm_storel_pd((double *)b11 + 2, _mm256_extractf128_pd(ymm3, 1));
      _mm_storel_pd((double *)(b11 + cs_b + 2), _mm256_extractf128_pd(ymm5, 1));
      _mm_storel_pd((double *)(b11 + (cs_b * 2) + 2), _mm256_extractf128_pd(ymm7, 1));
      _mm_storel_pd((double *)(b11 + (cs_b * 3) + 2), _mm256_extractf128_pd(ymm9, 1));

      m_remainder -= 3;
      i += 3;
    }
    else if (m_remainder == 2)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx2m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      xmm5 = _mm_loadu_pd((double const *)(b11 + (cs_b * 3)));
      ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
      ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
      // B11[0-3][3] * alpha -= ymm6

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      a11 += cs_a;

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

      _mm_storeu_pd((double *)b11, _mm256_castpd256_pd128(ymm3));
      _mm_storeu_pd((double *)(b11 + cs_b), _mm256_castpd256_pd128(ymm5));
      _mm_storeu_pd((double *)(b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
      _mm_storeu_pd((double *)(b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

      m_remainder -= 2;
      i += 2;
    }
    else if (m_remainder == 1)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx1m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_broadcast_sd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 3)));
      // B11[0][3] B11[1][3] B11[2][3] B11[3][3]
      ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
      // B11[0-3][3] * alpha -= ymm6

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm3, ymm9);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm5, ymm9);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      a11 += cs_a;

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * rs_a)));
      ymm9 = _mm256_fnmadd_pd(ymm1, ymm7, ymm9);

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

      _mm_storel_pd((b11 + (cs_b * 0)), _mm256_castpd256_pd128(ymm3));
      _mm_storel_pd((b11 + (cs_b * 1)), _mm256_castpd256_pd128(ymm5));
      _mm_storel_pd((b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
      _mm_storel_pd((b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

      m_remainder -= 1;
      i += 1;
    }
    j += 4;
    n_remainder -= 4;
  }

  if (n_remainder == 3)
  {
    a01 = L + j * rs_a;      // pointer to block of A to be used in GEMM
    a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = j; // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + p_lda * 0));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + p_lda * 1));
        bli_dcopys(*(a01 + rs_a * 2), *(ptr_a10_dup + p_lda * 2));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = p_lda / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 2) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 2) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 2) + 2));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 2) + 2));
      }
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = 0; (i + d_mr - 1) < m; i += d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));
      // B11[4][1] B11[5][1] B11[6][1] B11[7][1]

      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2
      ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);
      // B11[4-7][1] * alpha -= ymm3

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2) + 4));
      // B11[4][2] B11[5][2] B11[6][2] B11[7][2]

      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4
      ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);
      // B11[4-7][2] * alpha -= ymm5

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));

      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);
      ymm8 = _mm256_fnmadd_pd(ymm1, ymm4, ymm8);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));

      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);
      ymm8 = _mm256_fnmadd_pd(ymm1, ymm6, ymm8);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
      ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2) + 4), ymm8);
    }

    dim_t m_remainder = m - i;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      /// implement TRSM///
      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);

      m_remainder -= 4;
      i += 4;
    }

    if (m_remainder == 3)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx3m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      BLIS_POST_DTRSM_SMALL_3N_3M(b11, cs_b)

      m_remainder -= 3;
      i += 3;
    }
    else if (m_remainder == 2)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx2m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      BLIS_POST_DTRSM_SMALL_3N_2M(b11, cs_b)

      m_remainder -= 2;
      i += 2;
    }
    else if (m_remainder == 1)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx1m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm3, ymm7);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      a11 += cs_a;

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm5, ymm7);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      BLIS_POST_DTRSM_SMALL_3N_1M(b11, cs_b)

      m_remainder -= 1;
      i += 1;
    }
    j += 3;
    n_remainder -= 3;
  }
  else if (n_remainder == 2)
  {
    a01 = L + j * rs_a;      // pointer to block of A to be used in GEMM
    a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = j; // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + (p_lda * 0)));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + (p_lda * 1)));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = p_lda / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
      }
      ymm2 = _mm256_broadcast_sd((double const *)&ones);
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = 0; (i + d_mr - 1) < m; i += d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));
      // B11[4][1] B11[5][1] B11[6][1] B11[7][1]

      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2
      ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);
      // B11[4-7][1] * alpha -= ymm3

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm4, ymm6);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
    }

    dim_t m_remainder = m - i;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();
      ymm5 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);
      // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);

      m_remainder -= 4;
      i += 4;
    }

    if (m_remainder == 3)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();
      ymm5 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx3m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      BLIS_POST_DTRSM_SMALL_2N_3M(b11, cs_b)

      m_remainder -= 3;
      i += 3;
    }
    else if (m_remainder == 2)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();
      ymm5 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx2m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal, b11, cs_b)

      /// implement TRSM///
      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      BLIS_POST_DTRSM_SMALL_2N_2M(b11, cs_b)

      m_remainder -= 2;
      i += 2;
    }
    else if (m_remainder == 1)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();
      ymm5 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx1m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 1):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm3, ymm5);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      BLIS_POST_DTRSM_SMALL_2N_1M(b11, cs_b)

      m_remainder -= 1;
      i += 1;
    }
    j += 2;
    n_remainder -= 2;
  }
  else if (n_remainder == 1)
  {
    a01 = L + j * rs_a;      // pointer to block of A to be used in GEMM
    a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = j; // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + p_lda * 0));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = p_lda / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      // broadcast diagonal elements of A11
      ymm0 = _mm256_broadcast_sd((double const *)(a11));
      ymm1 = _mm256_broadcast_sd((double const *)&ones);
      ymm2 = _mm256_broadcast_sd((double const *)&ones);
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = 0; (i + d_mr - 1) < m; i += d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();
      ymm4 = _mm256_setzero_pd();
      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
    }

    dim_t m_remainder = m - i;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11); // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);   // B11[0-3][0] * alpha -= ymm0

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);

      m_remainder -= 4;
      i += 4;
    }

    if (m_remainder == 3)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx3m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      BLIS_POST_DTRSM_SMALL_1N_3M(b11, cs_b)

      m_remainder -= 3;
      i += 3;
    }
    else if (m_remainder == 2)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx2m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      BLIS_POST_DTRSM_SMALL_1N_2M(b11, cs_b)

      m_remainder -= 2;
      i += 2;
    }
    else if (m_remainder == 1)
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i;           // pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = j; // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx1m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      BLIS_POST_DTRSM_SMALL_1N_1M(b11, cs_b)

      m_remainder -= 1;
      i += 1;
    }
    j += 1;
    n_remainder -= 1;
  }

  if ((required_packing_A == 1) && bli_mem_is_alloc(&local_mem_buf_A_s))
  {
    bli_membrk_release(&rntm,
               &local_mem_buf_A_s);
  }
  return BLIS_SUCCESS;
}


// RLNN - RUTN
BLIS_INLINE err_t bli_dtrsm_small_XAutB_XAlB_AVX512
     (
       obj_t*   AlphaObj,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     )
{
  dim_t m = bli_obj_length(b); // number of rows
  dim_t n = bli_obj_width(b);  // number of columns
  dim_t d_mr = 8, d_nr = 8;

  bool transa = bli_obj_has_trans(a);
  dim_t cs_a, rs_a;
  double ones = 1.0;

  // Swap rs_a & cs_a in case of non-transpose.
  if (transa)
  {
    cs_a = bli_obj_col_stride(a); // column stride of A
    rs_a = bli_obj_row_stride(a); // row stride of A
  }
  else
  {
    cs_a = bli_obj_row_stride(a); // row stride of A
    rs_a = bli_obj_col_stride(a); // column stride of A
  }

  dim_t cs_b = bli_obj_col_stride(b); // column stride of B

  dim_t i, j, k;
  dim_t k_iter;

  bool is_unitdiag = bli_obj_has_unit_diag(a);

  double AlphaVal = *(double *)AlphaObj->buffer;
  double *restrict L = bli_obj_buffer_at_off(a); // pointer to matrix A
  double *B = bli_obj_buffer_at_off(b); // pointer to matrix B

  double *a01, *a11, *b10, *b11; // pointers for GEMM and TRSM blocks

  bool required_packing_A = true;
  mem_t local_mem_buf_A_s = {0};
  double *D_A_pack = NULL; // pointer to A01 pack buffer
  double d11_pack[d_mr] __attribute__((aligned(64))); // buffer for diagonal A pack
  rntm_t rntm;

  bli_rntm_init_from_global(&rntm);
  bli_rntm_set_num_threads_only(1, &rntm);
  bli_membrk_rntm_set_membrk(&rntm);

  siz_t buffer_size = bli_pool_block_size(
    bli_membrk_pool(
      bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
      bli_rntm_membrk(&rntm)));

  if ((d_nr * n * sizeof(double)) > buffer_size)
    return BLIS_NOT_YET_IMPLEMENTED;

  if (required_packing_A)
  {
    // Get the buffer from the pool.
    bli_membrk_acquire_m(&rntm,
               buffer_size,
               BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
               &local_mem_buf_A_s); // acquire memory for A01 pack
    if (FALSE == bli_mem_is_alloc(&local_mem_buf_A_s))
      return BLIS_NULL_POINTER;
    D_A_pack = bli_mem_buffer(&local_mem_buf_A_s);
    if (NULL == D_A_pack)
      return BLIS_NULL_POINTER;
  }
  __m512d zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11;
  __m512d zmm12, zmm13, zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
  __m512d zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;
  __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
  __m256d ymm12, ymm13, ymm14, ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21;
  __m256d ymm22, ymm23, ymm24, ymm25, ymm26, ymm27, ymm28, ymm29, ymm30, ymm31;
  __m128d xmm5, xmm0;

  /*
    Performs solving TRSM for 8 rows at a time from  0 to n/8 in steps of d_nr
    a. Load and pack A (a01 block), the size of packing 8x8 to 8x(n-8)
        First there will be no GEMM and no packing of a01 because it is only TRSM
    b. Using packed a01 block and b10 block perform GEMM operation
    c. Use GEMM outputs, perform TRSM operation using a11, b11 and update B
    d. Repeat b for m cols of B in steps of d_mr
  */
  for (j = (n - d_nr); j > -1; j -= d_nr) //loop along 'N' direction
  {
    a01 = L + (j * rs_a) + (j + d_nr) * cs_a;  //pointer to block of A to be used in GEMM
    a11 = L + (j * cs_a) + (j * rs_a);         //pointer to block of A to be used for TRSM

    dim_t p_lda = (n - j - d_nr);              //packed leading dimension

    // perform copy of A to packed buffer D_A_pack
    if (transa)
    {
      /*
      Pack current A block (a01) into packed buffer memory D_A_pack
        a. This a10 block is used in GEMM portion only and this
            a01 block size will be increasing by d_nr for every next iteration
            until it reaches 8x(n-8) which is the maximum GEMM alone block size in A
        b. This packed buffer is reused to calculate all m cols of B matrix
      */
      bli_dtrsm_small_pack_avx512
      (
        'R',
        p_lda,
        1,
        a01,
        cs_a,
        D_A_pack,
        p_lda,
        d_nr
      );
      /*
        Pack 8 diagonal elements of A block into an array
        a. This helps to utilize cache line efficiently in TRSM operation
        b. store ones when input is unit diagonal
      */
      dtrsm_small_pack_diag_element_avx512
      (
        is_unitdiag,
        a11,
        cs_a,
        d11_pack,
        d_nr
      );
    }
    else
    {
      bli_dtrsm_small_pack_avx512
      (
        'R',
        p_lda,
        0,
        a01,
        rs_a,
        D_A_pack,
        p_lda,
        d_nr
      );
      dtrsm_small_pack_diag_element_avx512
      (
        is_unitdiag,
        a11,
        rs_a,
        d11_pack,
        d_nr
      );
    }

    /*
      a. Perform GEMM using a01, b10.
      b. Perform TRSM on a11, b11
      c. This loop GEMM+TRSM loops operates with 8x6 block size
          along m dimension for every d_mr columns of B10 where
          packed A buffer is reused in computing all m cols of B.
      d. Same approach is used in remaining fringe cases.
    */
    for (i = (m - d_mr); (i + 1) > 0; i -= d_mr) //loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + j * cs_a + j * rs_a;   //pointer to block of A to be used for TRSM
      b10 = B + i + (j + d_nr) * cs_b; //pointer to block of B to be used in GEMM
      b11 = B + i + j * cs_b;          //pointer to block of B to be used for TRSM

      k_iter = (n - j - d_nr);
      BLIS_SET_ZMM_REG_ZEROS
      /*
        Perform GEMM between a01 and b10 blocks
        For first iteration there will be no GEMM operation
        where k_iter are zero
      */
      BLIS_DTRSM_SMALL_GEMM_8nx8m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11);
      /*
        Load b11 of size 8x8 and multiply with alpha
        Add the GEMM output to b11
        and perform TRSM operation.
      */
      BLIS_PRE_DTRSM_SMALL_8x8(AlphaVal, b11, cs_b)



      /*
        Compute 8x8 TRSM block by using GEMM block output in register
        a. The 8x8 input (gemm outputs) are stored in combinations of zmm registers
            row      :   0     1    2      3     4     5     6     7
            register : zmm9  zmm10 zmm11 zmm12 zmm13 zmm14 zmm15 zmm16
        b. Towards the end TRSM output will be stored back into b11
      */

      /*
      *                                         to i=7
      *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[N][i]  } /A11[N][N]
      *                                       from i=n+1
      *
      *  For example 3rd column (B11[2]) -= ((B11[3] * A11[2][3]) + (B11[4] * A11[2][4]) +
      *                                      (B11[5] * A11[2][5]) + (B11[6] * A11[2][6]) +
      *                                      (B11[7] * A11[2][7])) / A11[2][2]
      *                          zmm11   -= ((zmm12  * A11[2][3]) + (zmm13  * A11[2][4]) +
      *                                      (zmm14  * A11[2][5]) + (zmm15  * A11[2][6]) +
      *                                      (zmm16  * A11[2][7])) / A11[2][2]
      */

      // extract a77
      zmm0  = _mm512_set1_pd(*(d11_pack + 7));
      zmm16 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm16, zmm0);
      _mm512_storeu_pd((double *)(b11 + 7 * cs_b), zmm16);

      // extract a66
      zmm0  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (6 * rs_a)));
      zmm1  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (5 * rs_a)));
      zmm15 = _mm512_fnmadd_pd(zmm0, zmm16, zmm15);
      zmm0  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (4 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm1, zmm16, zmm14);
      zmm1  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (3 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm0, zmm16, zmm13);
      zmm0  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (2 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm1, zmm16, zmm12);
      zmm1  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (1 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm0, zmm16, zmm11);
      zmm0  = _mm512_set1_pd(*(a11 + (7 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm16, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 6));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm16, zmm9);
      zmm15 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm15, zmm1);
      _mm512_storeu_pd((double *)(b11 + (6 * cs_b)), zmm15);

      // extract a55
      zmm1  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (5 * rs_a)));
      zmm0  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (4 * rs_a)));
      zmm14 = _mm512_fnmadd_pd(zmm1, zmm15, zmm14);
      zmm1  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (3 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm0, zmm15, zmm13);
      zmm0  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (2 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm1, zmm15, zmm12);
      zmm1  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (1 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm0, zmm15, zmm11);
      zmm0  = _mm512_set1_pd(*(a11 + (6 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm15, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 5));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm15, zmm9);
      zmm14 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm14, zmm1);
      _mm512_storeu_pd((double *)(b11 + (5 * cs_b)), zmm14);

      // extract a44
      zmm0  = _mm512_set1_pd(*(a11 + (5 * cs_a) + (4 * rs_a)));
      zmm1  = _mm512_set1_pd(*(a11 + (5 * cs_a) + (3 * rs_a)));
      zmm13 = _mm512_fnmadd_pd(zmm0, zmm14, zmm13);
      zmm0  = _mm512_set1_pd(*(a11 + (5 * cs_a) + (2 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm1, zmm14, zmm12);
      zmm1  = _mm512_set1_pd(*(a11 + (5 * cs_a) + (1 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm0, zmm14, zmm11);
      zmm0  = _mm512_set1_pd(*(a11 + (5 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm14, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 4));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm14, zmm9);
      zmm13 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm13, zmm1);
      _mm512_storeu_pd((double *)(b11 + (4 * cs_b)), zmm13);

      // extract a33
      zmm1  = _mm512_set1_pd(*(a11 + (4 * cs_a) + (3 * rs_a)));
      zmm0  = _mm512_set1_pd(*(a11 + (4 * cs_a) + (2 * rs_a)));
      zmm12 = _mm512_fnmadd_pd(zmm1, zmm13, zmm12);
      zmm1  = _mm512_set1_pd(*(a11 + (4 * cs_a) + (1 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm0, zmm13, zmm11);
      zmm0  = _mm512_set1_pd(*(a11 + (4 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm13, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 3));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm13, zmm9);
      zmm12 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm12, zmm1);
      _mm512_storeu_pd((double *)(b11 + (3 * cs_b)), zmm12);

      // extract a22
      zmm0  = _mm512_set1_pd(*(a11 + (3 * cs_a) + (2 * rs_a)));
      zmm1  = _mm512_set1_pd(*(a11 + (3 * cs_a) + (1 * rs_a)));
      zmm11 = _mm512_fnmadd_pd(zmm0, zmm12, zmm11);
      zmm0  = _mm512_set1_pd(*(a11 + (3 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm12, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 2));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm12, zmm9);
      zmm11 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm11, zmm1);
      _mm512_storeu_pd((double *)(b11 + (2 * cs_b)), zmm11);

      // extract a11
      zmm1  = _mm512_set1_pd(*(a11 + (2 * cs_a) + (1 * rs_a)));
      zmm0  = _mm512_set1_pd(*(a11 + (2 * cs_a) + (0 * rs_a)));
      zmm10 = _mm512_fnmadd_pd(zmm1, zmm11, zmm10);
      zmm1  = _mm512_set1_pd(*(d11_pack + 1));
      zmm9  = _mm512_fnmadd_pd(zmm0, zmm11, zmm9);
      zmm10 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm10, zmm1);
      _mm512_storeu_pd((double *)(b11 + (1 * cs_b)), zmm10);

      // extract a00
      zmm1 = _mm512_set1_pd(*(a11 + (1 * cs_a) + (0 * rs_a)));
      zmm0 = _mm512_set1_pd(*(d11_pack + 0));
      zmm9 = _mm512_fnmadd_pd(zmm1, zmm10, zmm9);
      zmm9 = DTRSM_SMALL_DIV_OR_SCALE_AVX512(zmm9, zmm0);
      _mm512_storeu_pd((double *)(b11 + (0 * cs_b)), zmm9);
    }
    dim_t m_remainder = i + d_mr;
    if(m_remainder)
    {
      if (m_remainder >= 4) //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + (j * cs_a) + (j * rs_a);               //pointer to block of A to be used for TRSM
        b10 = B + (m_remainder - 4) + (j + d_nr) * cs_b; //pointer to block of B to be used in GEMM
        b11 = B + (m_remainder - 4) + (j * cs_b);        //pointer to block of B to be used for TRSM

        k_iter = (n - j - d_nr);
        BLIS_SET_YMM_REG_ZEROS_AVX512
        /*
          Perform GEMM between a01 and b10 blocks
          For first iteration there will be no GEMM operation
          where k_iter are zero
        */
        BLIS_DTRSM_SMALL_GEMM_8nx4m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        /*
          Load b11 of size 8x4 and multiply with alpha
          Add the GEMM output to b11
          and perform TRSM operation.
        */
        BLIS_PRE_DTRSM_SMALL_8x4(AlphaVal, b11, cs_b)

        /*
          Compute 8x4 TRSM block by using GEMM block output in register
          a. The 8x4 input (gemm outputs) are stored in combinations of ymm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                         to i=7
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[N][i]  } /A11[N][N]
        *                                       from i=n+1
        *
        *  For example 3rd column (B11[2]) -= ((B11[3] * A11[2][3]) + (B11[4] * A11[2][4]) +
        *                                      (B11[5] * A11[2][5]) + (B11[6] * A11[2][6]) +
        *                                      (B11[7] * A11[2][7])) / A11[2][2]
        */

        // extract a77
        ymm0  = _mm256_broadcast_sd((d11_pack + 7));
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm0);
        _mm256_storeu_pd((double *)(b11 + (7 * cs_b)), ymm16);

        // extract a66
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (6 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (5 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm0, ymm16, ymm15);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm16, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm16, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm16, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm16, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm16, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 6));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm16, ymm9);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
        _mm256_storeu_pd((double *)(b11 + (6 * cs_b)), ymm15);

        // extract a55
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (5 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm15, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm15, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm15, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm15, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm15, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 5));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm15, ymm9);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        _mm256_storeu_pd((double *)(b11 + (5 * cs_b)), ymm14);

        // extract a44
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (4 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm14, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm14, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm14, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm14, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 4));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm14, ymm9);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
        _mm256_storeu_pd((double *)(b11 + (4 * cs_b)), ymm13);

        // extract a33
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (3 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm13, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm13, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm13, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 3));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm13, ymm9);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        _mm256_storeu_pd((double *)(b11 + (3 * cs_b)), ymm12);

        // extract a22
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (2 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm12, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm12, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 2));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm12, ymm9);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
        _mm256_storeu_pd((double *)(b11 + (2 * cs_b)), ymm11);

        // extract a11
        ymm1  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (1 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm11, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 1));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm11, ymm9);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        _mm256_storeu_pd((double *)(b11 + (1 * cs_b)), ymm10);

        // extract a00
        ymm1 = _mm256_broadcast_sd((a11 + (1 * cs_a) + (0 * rs_a)));
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = _mm256_fnmadd_pd(ymm1, ymm10, ymm9);
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm256_storeu_pd((double *)(b11 + (0 * cs_b)), ymm9);
        m_remainder -= 4;
      }
      if (m_remainder == 3) //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + (j * cs_a) + (j * rs_a);               //pointer to block of A to be used for TRSM
        b10 = B + (j + d_nr) * cs_b; // pointer to block of B to be used in GEMM
        b11 = B + (j * cs_b);        //pointer to block of B to be used for TRSM

        k_iter = (n - j - d_nr);
        BLIS_SET_YMM_REG_ZEROS_AVX512
        /*
          Perform GEMM between a01 and b10 blocks
          For first iteration there will be no GEMM operation
          where k_iter are zero
        */
        BLIS_DTRSM_SMALL_GEMM_8nx3m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        /*
          Load b11 of size 8x3 and multiply with alpha
          Add the GEMM output to b11
          and perform TRSM operation.
        */
        BLIS_PRE_DTRSM_SMALL_8x3(AlphaVal, b11, cs_b)
        /*
          Compute 8x3 TRSM block by using GEMM block output in register
          a. The 8x3 input (gemm outputs) are stored in combinations of ymm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                         to i=7
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[N][i]  } /A11[N][N]
        *                                       from i=n+1
        *
        *  For example 3rd column (B11[2]) -= ((B11[3] * A11[2][3]) + (B11[4] * A11[2][4]) +
        *                                      (B11[5] * A11[2][5]) + (B11[6] * A11[2][6]) +
        *                                      (B11[7] * A11[2][7])) / A11[2][2]
        */

        // extract a77
        ymm0  = _mm256_broadcast_sd((d11_pack + 7));
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm0);
        _mm_storeu_pd((double *)(b11 + (7 * cs_b) + 0), _mm256_castpd256_pd128(ymm16));
        _mm_storel_pd((double *)(b11 + (7 * cs_b) + 2), _mm256_extractf64x2_pd(ymm16, 1));

        // extract a66
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (6 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (5 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm0, ymm16, ymm15);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm16, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm16, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm16, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm16, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm16, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 6));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm16, ymm9);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
        _mm_storeu_pd((double *)(b11 + (6 * cs_b) + 0), _mm256_castpd256_pd128(ymm15));
        _mm_storel_pd((double *)(b11 + (6 * cs_b) + 2), _mm256_extractf64x2_pd(ymm15, 1));

        // extract a55
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (5 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm15, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm15, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm15, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm15, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm15, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 5));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm15, ymm9);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        _mm_storeu_pd((double *)(b11 + (5 * cs_b) + 0), _mm256_castpd256_pd128(ymm14));
        _mm_storel_pd((double *)(b11 + (5 * cs_b) + 2), _mm256_extractf64x2_pd(ymm14, 1));

        // extract a44
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (4 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm14, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm14, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm14, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm14, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 4));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm14, ymm9);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
        _mm_storeu_pd((double *)(b11 + (4 * cs_b) + 0), _mm256_castpd256_pd128(ymm13));
        _mm_storel_pd((double *)(b11 + (4 * cs_b) + 2), _mm256_extractf64x2_pd(ymm13, 1));

        // extract a33
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (3 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm13, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm13, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm13, ymm10);
        ymm1 = _mm256_broadcast_sd((d11_pack + 3));
        ymm9 = _mm256_fnmadd_pd(ymm0, ymm13, ymm9);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        _mm_storeu_pd((double *)(b11 + (3 * cs_b) + 0), _mm256_castpd256_pd128(ymm12));
        _mm_storel_pd((double *)(b11 + (3 * cs_b) + 2), _mm256_extractf64x2_pd(ymm12, 1));

        // extract a22
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (2 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm12, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm12, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 2));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm12, ymm9);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
        _mm_storeu_pd((double *)(b11 + (2 * cs_b) + 0), _mm256_castpd256_pd128(ymm11));
        _mm_storel_pd((double *)(b11 + (2 * cs_b) + 2), _mm256_extractf64x2_pd(ymm11, 1));

        // extract a11
        ymm1  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (1 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm11, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 1));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm11, ymm9);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        _mm_storeu_pd((double *)(b11 + (1 * cs_b) + 0), _mm256_castpd256_pd128(ymm10));
        _mm_storel_pd((double *)(b11 + (1 * cs_b) + 2), _mm256_extractf64x2_pd(ymm10, 1));

        // extract a00
        ymm1 = _mm256_broadcast_sd((a11 + (1 * cs_a) + (0 * rs_a)));
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = _mm256_fnmadd_pd(ymm1, ymm10, ymm9);
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm_storeu_pd((double *)(b11 + (0 * cs_b) + 0), _mm256_castpd256_pd128(ymm9));
        _mm_storel_pd((double *)(b11 + (0 * cs_b) + 2), _mm256_extractf64x2_pd(ymm9, 1));
        m_remainder -= 3;
      }
      else if (m_remainder == 2) //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + (j * cs_a) + (j * rs_a);               //pointer to block of A to be used for TRSM
        b10 = B + (j + d_nr) * cs_b; // pointer to block of B to be used in GEMM
        b11 = B + (j * cs_b);        //pointer to block of B to be used for TRSM

        k_iter = (n - j - d_nr);
        BLIS_SET_YMM_REG_ZEROS_AVX512
        BLIS_DTRSM_SMALL_GEMM_8nx2m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11)
        BLIS_PRE_DTRSM_SMALL_8x2(AlphaVal, b11, cs_b)
        /*
          Compute 8x2 TRSM block by using GEMM block output in register
          a. The 8x2 input (gemm outputs) are stored in combinations of zmm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                         to i=7
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[N][i]  } /A11[N][N]
        *                                       from i=n+1
        *
        *  For example 3rd column (B11[2]) -= ((B11[3] * A11[2][3]) + (B11[4] * A11[2][4]) +
        *                                      (B11[5] * A11[2][5]) + (B11[6] * A11[2][6]) +
        *                                      (B11[7] * A11[2][7])) / A11[2][2]
        */

        // extract a77
        ymm0  = _mm256_broadcast_sd((d11_pack + 7));
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm0);
        _mm_storeu_pd((double *)(b11 + (7 * cs_b)), _mm256_castpd256_pd128(ymm16));

        // extract a66
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (6 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (5 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm0, ymm16, ymm15);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm16, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm16, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm16, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm16, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm16, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 6));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm16, ymm9);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
        _mm_storeu_pd((double *)(b11 + (6 * cs_b)), _mm256_castpd256_pd128(ymm15));

        // extract a55
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (5 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm15, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm15, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm15, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm15, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm15, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 5));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm15, ymm9);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        _mm_storeu_pd((double *)(b11 + (5 * cs_b)), _mm256_castpd256_pd128(ymm14));

        // extract a44
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (4 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm14, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm14, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm14, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm14, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 4));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm14, ymm9);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
        _mm_storeu_pd((double *)(b11 + (4 * cs_b)), _mm256_castpd256_pd128(ymm13));

        // extract a33
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (3 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm13, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm13, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm13, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 3));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm13, ymm9);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        _mm_storeu_pd((double *)(b11 + (3 * cs_b)), _mm256_castpd256_pd128(ymm12));

        // extract a22
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (2 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm12, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm12, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 2));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm12, ymm9);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
        _mm_storeu_pd((double *)(b11 + (2 * cs_b)), _mm256_castpd256_pd128(ymm11));

        // extract a11
        ymm1  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (1 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm11, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 1));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm11, ymm9);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        _mm_storeu_pd((double *)(b11 + (1 * cs_b)), _mm256_castpd256_pd128(ymm10));

        // extract a00
        ymm1 = _mm256_broadcast_sd((a11 + (1 * cs_a) + (0 * rs_a)));
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = _mm256_fnmadd_pd(ymm1, ymm10, ymm9);
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm_storeu_pd((double *)(b11 + (0 * cs_b)), _mm256_castpd256_pd128(ymm9));
        m_remainder -= 2;
      }
      else if (m_remainder == 1)  //loop along 'M' direction
      {
        a01 = D_A_pack;
        a11 = L + (j * cs_a) + (j * rs_a);               //pointer to block of A to be used for TRSM
        b10 = B + (j + d_nr) * cs_b; //pointer to block of B to be used in GEMM
        b11 = B + (j * cs_b);        //pointer to block of B to be used for TRSM

        k_iter = (n - j - d_nr);
        BLIS_SET_YMM_REG_ZEROS_AVX512
        BLIS_DTRSM_SMALL_GEMM_8nx1m_AVX512(a01, b10, cs_b, p_lda, k_iter, b11);
        BLIS_PRE_DTRSM_SMALL_8x1(AlphaVal, b11, cs_b)
        /*
          Compute 8x1 TRSM block by using GEMM block output in register
          a. The 8x1 input (gemm outputs) are stored in combinations of zmm registers
              row      :   0     1    2      3     4     5     6     7
              register : ymm9  ymm10 ymm11 ymm12 ymm13 ymm14 ymm15 ymm16
          b. Towards the end TRSM output will be stored back into b11
        */

        /*
        *                                         to i=7
        *  B11[Nth column] = GEMM(Nth column) -     Σ  {  B11[i] * A11[N][i]  } /A11[N][N]
        *                                       from i=n+1
        *
        *  For example 3rd column (B11[2]) -= ((B11[3] * A11[2][3]) + (B11[4] * A11[2][4]) +
        *                                      (B11[5] * A11[2][5]) + (B11[6] * A11[2][6]) +
        *                                      (B11[7] * A11[2][7])) / A11[2][2]
        */

        // extract a77
        ymm0  = _mm256_broadcast_sd((d11_pack + 7));
        ymm16 = DTRSM_SMALL_DIV_OR_SCALE(ymm16, ymm0);
        _mm_storel_pd((double *)(b11 + (7 * cs_b)), _mm256_castpd256_pd128(ymm16));

        // extract a66
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (6 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (5 * rs_a)));
        ymm15 = _mm256_fnmadd_pd(ymm0, ymm16, ymm15);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm16, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm16, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm16, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm16, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (7 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm16, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 6));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm16, ymm9);
        ymm15 = DTRSM_SMALL_DIV_OR_SCALE(ymm15, ymm1);
        _mm_storel_pd((double *)(b11 + (6 * cs_b)), _mm256_castpd256_pd128(ymm15));

        // extract a55
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (5 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (4 * rs_a)));
        ymm14 = _mm256_fnmadd_pd(ymm1, ymm15, ymm14);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm15, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm15, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm15, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (6 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm15, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 5));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm15, ymm9);
        ymm14 = DTRSM_SMALL_DIV_OR_SCALE(ymm14, ymm1);
        _mm_storel_pd((double *)(b11 + (5 * cs_b)), _mm256_castpd256_pd128(ymm14));

        // extract a44
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (4 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (3 * rs_a)));
        ymm13 = _mm256_fnmadd_pd(ymm0, ymm14, ymm13);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm14, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm14, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (5 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm14, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 4));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm14, ymm9);
        ymm13 = DTRSM_SMALL_DIV_OR_SCALE(ymm13, ymm1);
        _mm_storel_pd((double *)(b11 + (4 * cs_b)), _mm256_castpd256_pd128(ymm13));

        // extract a33
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (3 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (2 * rs_a)));
        ymm12 = _mm256_fnmadd_pd(ymm1, ymm13, ymm12);
        ymm1  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm13, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (4 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm13, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 3));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm13, ymm9);
        ymm12 = DTRSM_SMALL_DIV_OR_SCALE(ymm12, ymm1);
        _mm_storel_pd((double *)(b11 + (3 * cs_b)), _mm256_castpd256_pd128(ymm12));

        // extract a22
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (2 * rs_a)));
        ymm1  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (1 * rs_a)));
        ymm11 = _mm256_fnmadd_pd(ymm0, ymm12, ymm11);
        ymm0  = _mm256_broadcast_sd((a11 + (3 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm12, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 2));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm12, ymm9);
        ymm11 = DTRSM_SMALL_DIV_OR_SCALE(ymm11, ymm1);
        _mm_storel_pd((double *)(b11 + (2 * cs_b)), _mm256_castpd256_pd128(ymm11));

        // extract a11
        ymm1  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (1 * rs_a)));
        ymm0  = _mm256_broadcast_sd((a11 + (2 * cs_a) + (0 * rs_a)));
        ymm10 = _mm256_fnmadd_pd(ymm1, ymm11, ymm10);
        ymm1  = _mm256_broadcast_sd((d11_pack + 1));
        ymm9  = _mm256_fnmadd_pd(ymm0, ymm11, ymm9);
        ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm1);
        _mm_storel_pd((double *)(b11 + (1 * cs_b)), _mm256_castpd256_pd128(ymm10));

        // extract a00
        ymm1 = _mm256_broadcast_sd((a11 + (1 * cs_a) + (0 * rs_a)));
        ymm0 = _mm256_broadcast_sd((d11_pack + 0));
        ymm9 = _mm256_fnmadd_pd(ymm1, ymm10, ymm9);
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
        _mm_storel_pd((double *)(b11 + (0 * cs_b)), _mm256_castpd256_pd128(ymm9));
        m_remainder -= 1;
    }
    }
  }

  dim_t n_remainder = j + d_nr;

  /*
  Reminder cases starts here:
  a. Similar logic and code flow used in computing full block (8x8)
     above holds for reminder cases too.
  */

  if (n_remainder >= 4)
  {
    a01 = L + (n_remainder - 4) * rs_a + n_remainder * cs_a;     // pointer to block of A to be used in GEMM
    a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = (n - n_remainder); // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + (p_lda * 0)));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + (p_lda * 1)));
        bli_dcopys(*(a01 + rs_a * 2), *(ptr_a10_dup + (p_lda * 2)));
        bli_dcopys(*(a01 + rs_a * 3), *(ptr_a10_dup + (p_lda * 3)));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = (n - n_remainder) / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 2) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 3) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 3) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 2) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 3) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 3) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    // read diagonal from a11 if not unit diagonal
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 2) + 2));
        ymm3 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 3) + 3));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 2) + 2));
        ymm3 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 3) + 3));
      }

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = (m - d_mr); (i + 1) > 0; i -= d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + i + (n_remainder)*cs_b;                // pointer to block of B to be used in GEMM
      b11 = B + (i) + (n_remainder - 4) * cs_b;            // pointer to block of B to be used for TRSM

      k_iter = (n - n_remainder); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx8m(a01, b10, cs_b, p_lda, k_iter)

      BLIS_PRE_DTRSM_SMALL_4x8(AlphaVal, b11, cs_b)

      /// implement TRSM///

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));

      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);
      ymm10 = DTRSM_SMALL_DIV_OR_SCALE(ymm10, ymm0);

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (2 * rs_a)));

      ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);
      ymm8 = _mm256_fnmadd_pd(ymm1, ymm10, ymm8);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm10, ymm6);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a)));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm10, ymm4);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
      ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm8, ymm6);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm8, ymm4);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b + 4)), ymm6);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2) + 4), ymm8);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3)), ymm9);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3) + 4), ymm10);
    }

    dim_t m_remainder = i + d_mr;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM
      b10 = B + (m_remainder - 4) + (n_remainder)*cs_b;        // pointer to block of B to be used in GEMM
      b11 = B + (m_remainder - 4) + (n_remainder - 4) * cs_b;    // pointer to block of B to be used for TRSM

      k_iter = (n - n_remainder); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_4nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)(&AlphaVal)); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 3)));
      // B11[0][3] B11[1][3] B11[2][3] B11[3][3]
      ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
      // B11[0-3][3] * alpha -= ymm6

      /// implement TRSM///

      // extract a33
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
      ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      //(Row 3): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (2 * rs_a)));
      ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a)));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 3)), ymm9);

      m_remainder -= 4;
    }

    if (m_remainder)
    {
      if (m_remainder == 3)
      {
        a01 = D_A_pack;
        a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM
        b10 = B + (n_remainder)*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B + (n_remainder - 4) * cs_b;    // pointer to block of B to be used for TRSM

        k_iter = (n - n_remainder); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_4nx3m(a01, b10, cs_b, p_lda, k_iter)

        ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

        ymm0 = _mm256_loadu_pd((double const *)b11);
        // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
        ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
        // B11[0-3][0] * alpha -= ymm0

        ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
        // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
        ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
        // B11[0-3][1] * alpha-= ymm2

        ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
        // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
        ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
        // B11[0-3][2] * alpha -= ymm4

        xmm5 = _mm_loadu_pd((double const *)(b11 + (cs_b * 3)));
        ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 3) + 2));
        ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
        ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
        // B11[0-3][3] * alpha -= ymm6

        /// implement TRSM///

        // extract a33
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

        //(Row 3): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (2 * rs_a)));
        ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        _mm_storeu_pd((double *)b11, _mm256_castpd256_pd128(ymm3));
        _mm_storeu_pd((double *)(b11 + cs_b), _mm256_castpd256_pd128(ymm5));
        _mm_storeu_pd((double *)(b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
        _mm_storeu_pd((double *)(b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

        _mm_storel_pd((double *)b11 + 2, _mm256_extractf128_pd(ymm3, 1));
        _mm_storel_pd((double *)(b11 + cs_b + 2), _mm256_extractf128_pd(ymm5, 1));
        _mm_storel_pd((double *)(b11 + (cs_b * 2) + 2), _mm256_extractf128_pd(ymm7, 1));
        _mm_storel_pd((double *)(b11 + (cs_b * 3) + 2), _mm256_extractf128_pd(ymm9, 1));

        m_remainder -= 3;
      }
      else if (m_remainder == 2)
      {
        a01 = D_A_pack;
        a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM
        b10 = B + (n_remainder)*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B + (n_remainder - 4) * cs_b;    // pointer to block of B to be used for TRSM

        k_iter = (n - n_remainder); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_4nx2m(a01, b10, cs_b, p_lda, k_iter)

        ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

        ymm0 = _mm256_loadu_pd((double const *)b11);
        // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
        ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
        // B11[0-3][0] * alpha -= ymm0

        ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
        // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
        ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
        // B11[0-3][1] * alpha-= ymm2

        ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
        // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
        ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
        // B11[0-3][2] * alpha -= ymm4

        xmm5 = _mm_loadu_pd((double const *)(b11 + (cs_b * 3)));
        ymm0 = _mm256_insertf128_pd(ymm0, xmm5, 0);
        ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
        // B11[0-3][3] * alpha -= ymm6

        /// implement TRSM///

        // extract a33
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

        //(Row 3): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (2 * rs_a)));
        ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        _mm_storeu_pd((double *)b11, _mm256_castpd256_pd128(ymm3));
        _mm_storeu_pd((double *)(b11 + cs_b), _mm256_castpd256_pd128(ymm5));
        _mm_storeu_pd((double *)(b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
        _mm_storeu_pd((double *)(b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

        m_remainder -= 2;
      }
      else if (m_remainder == 1)
      {
        a01 = D_A_pack;
        a11 = L + (n_remainder - 4) * cs_a + (n_remainder - 4) * rs_a; // pointer to block of A to be used for TRSM
        b10 = B + (n_remainder)*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B + (n_remainder - 4) * cs_b;    // pointer to block of B to be used for TRSM

        k_iter = (n - n_remainder); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_4nx1m(a01, b10, cs_b, p_lda, k_iter)

        ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);// register to hold alpha

        ymm0 = _mm256_broadcast_sd((double const *)b11);
        // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
        ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
        // B11[0-3][0] * alpha -= ymm0

        ymm0 = _mm256_broadcast_sd((double const *)(b11 + cs_b));
        // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
        ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
        // B11[0-3][1] * alpha-= ymm2

        ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 2)));
        // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
        ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
        // B11[0-3][2] * alpha -= ymm4

        ymm0 = _mm256_broadcast_sd((double const *)(b11 + (cs_b * 3)));
        // B11[0][3] B11[1][3] B11[2][3] B11[3][3]
        ymm9 = _mm256_fmsub_pd(ymm0, ymm15, ymm9);
        // B11[0-3][3] * alpha -= ymm6

        /// implement TRSM///

        // extract a33
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 3));
        ymm9 = DTRSM_SMALL_DIV_OR_SCALE(ymm9, ymm0);

        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

        //(Row 3): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (2 * rs_a)));
        ymm7 = _mm256_fnmadd_pd(ymm1, ymm9, ymm7);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm9, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (3 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm9, ymm3);

        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        _mm_storel_pd((b11 + (cs_b * 0)), _mm256_castpd256_pd128(ymm3));
        _mm_storel_pd((b11 + (cs_b * 1)), _mm256_castpd256_pd128(ymm5));
        _mm_storel_pd((b11 + (cs_b * 2)), _mm256_castpd256_pd128(ymm7));
        _mm_storel_pd((b11 + (cs_b * 3)), _mm256_castpd256_pd128(ymm9));

        m_remainder -= 1;
      }
    }
    n_remainder -= 4;
  }

  if (n_remainder == 3)
  {
    a01 = L + 3*cs_a;     // pointer to block of A to be used in GEMM
    a11 = L; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = (n - 3); // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + p_lda * 0));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + p_lda * 1));
        bli_dcopys(*(a01 + rs_a * 2), *(ptr_a10_dup + p_lda * 2));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = (n - 3) / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 2) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 2) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 2) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 2) + 2));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
        ymm2 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 2) + 2));
      }
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = (m - d_mr); (i + 1) > 0; i -= d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + i + 3*cs_b;                // pointer to block of B to be used in GEMM
      b11 = B + i;            // pointer to block of B to be used for TRSM

      k_iter = (n - 3); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));
      // B11[4][1] B11[5][1] B11[6][1] B11[7][1]

      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2
      ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);
      // B11[4-7][1] * alpha -= ymm3

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2) + 4));
      // B11[4][2] B11[5][2] B11[6][2] B11[7][2]

      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4
      ymm8 = _mm256_fmsub_pd(ymm1, ymm15, ymm8);
      // B11[4-7][2] * alpha -= ymm5

      /// implement TRSM///

      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));

      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);
      ymm8 = DTRSM_SMALL_DIV_OR_SCALE(ymm8, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));

      ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);
      ymm6 = _mm256_fnmadd_pd(ymm1, ymm8, ymm6);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm8, ymm4);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2) + 4), ymm8);
    }

    dim_t m_remainder = i + d_mr;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + (m_remainder - 4) + 3*cs_b;        // pointer to block of B to be used in GEMM
      b11 = B + (m_remainder - 4);   // pointer to block of B to be used for TRSM

      k_iter = (n - 3); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_3nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      ymm0 = _mm256_loadu_pd((double const *)(b11 + (cs_b * 2)));
      // B11[0][2] B11[1][2] B11[2][2] B11[3][2]
      ymm7 = _mm256_fmsub_pd(ymm0, ymm15, ymm7);
      // B11[0-3][2] * alpha -= ymm4

      /// implement TRSM///
      // extract a22
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
      ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      //(row 2):FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
      ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

      ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + (cs_b * 2)), ymm7);

      m_remainder -= 4;
    }

    if (m_remainder)
    {
      if (m_remainder == 3)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 3*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 3); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_3nx3m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_3N_3M(AlphaVal, b11, cs_b)

        /// implement TRSM///
        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_3N_3M(b11, cs_b)

        m_remainder -= 3;
      }
      else if (m_remainder == 2)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 3*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 3); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_3nx2m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_3N_2M(AlphaVal, b11, cs_b)

        /// implement TRSM///

        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_3N_2M(b11, cs_b)

        m_remainder -= 2;
      }
      else if (m_remainder == 1)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 3*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 3); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_3nx1m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_3N_1M(AlphaVal, b11, cs_b)

        /// implement TRSM///

        // extract a22
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 2));
        ymm7 = DTRSM_SMALL_DIV_OR_SCALE(ymm7, ymm0);

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

        //(row 2):FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a) + (1 * rs_a)));
        ymm5 = _mm256_fnmadd_pd(ymm1, ymm7, ymm5);

        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (2 * cs_a)));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm7, ymm3);

        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_3N_1M(b11, cs_b)

        m_remainder -= 1;
      }
    }
    n_remainder -= 3;
  }
else if ( n_remainder == 2)
  {
    a01 = L + 2*cs_a;     // pointer to block of A to be used in GEMM
    a11 = L; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = (n - 2); // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01 + rs_a * 0), *(ptr_a10_dup + (p_lda * 0)));
        bli_dcopys(*(a01 + rs_a * 1), *(ptr_a10_dup + (p_lda * 1)));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = (n - 2) / 4;

      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 1) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 1) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 1) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      if (transa)
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (cs_a * 1) + 1));
      }
      else
      {
        // broadcast diagonal elements of A11
        ymm0 = _mm256_broadcast_sd((double const *)(a11));
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + (rs_a * 1) + 1));
      }
      ymm2 = _mm256_broadcast_sd((double const *)&ones);
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = (m - d_mr); (i + 1) > 0; i -= d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + i + 2*cs_b;                // pointer to block of B to be used in GEMM
      b11 = B + i;            // pointer to block of B to be used for TRSM

      k_iter = (n - 2); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      BLIS_SET_YMM_REG_ZEROS_FOR_N_REM

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + cs_b + 4));
      // B11[4][1] B11[5][1] B11[6][1] B11[7][1]

      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2
      ymm6 = _mm256_fmsub_pd(ymm1, ymm15, ymm6);
      // B11[4-7][1] * alpha -= ymm3

      /// implement TRSM///

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));

      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);
      ymm6 = DTRSM_SMALL_DIV_OR_SCALE(ymm6, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));

      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);
      ymm4 = _mm256_fnmadd_pd(ymm1, ymm6, ymm4);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);
      _mm256_storeu_pd((double *)(b11 + cs_b + 4), ymm6);
    }

    dim_t m_remainder = i + d_mr;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + (m_remainder - 4) + 2*cs_b;        // pointer to block of B to be used in GEMM
      b11 = B + (m_remainder - 4);    // pointer to block of B to be used for TRSM

      k_iter = (n - 2); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      ymm3 = _mm256_setzero_pd();
      ymm5 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_2nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);
      // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      ymm0 = _mm256_loadu_pd((double const *)(b11 + cs_b));
      // B11[0][1] B11[1][1] B11[2][1] B11[3][1]
      ymm5 = _mm256_fmsub_pd(ymm0, ymm15, ymm5);
      // B11[0-3][1] * alpha-= ymm2

      /// implement TRSM///

      // extract a11
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
      ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

      //(Row 1): FMA operations
      ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
      ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + cs_b), ymm5);

      m_remainder -= 4;
    }

    if (m_remainder)
    {
      if (m_remainder == 3)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 2*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 2); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        ymm3 = _mm256_setzero_pd();
        ymm5 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_2nx3m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_2N_3M(AlphaVal, b11, cs_b)

        /// implement TRSM///

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_2N_3M(b11, cs_b)

        m_remainder -= 3;
      }
      else if (m_remainder == 2)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + (2)*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;     // pointer to block of B to be used for TRSM

        k_iter = (n -  2); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        ymm3 = _mm256_setzero_pd();
        ymm5 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_2nx2m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_2N_2M(AlphaVal, b11, cs_b)
        /// implement TRSM///

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_2N_2M(b11, cs_b)

        m_remainder -= 2;
      }
      else if (m_remainder == 1)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 2*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 2); // number of GEMM operations to be done(in blocks of 4x4)

        /*Fill zeros into ymm registers used in gemm accumulations */
        ymm3 = _mm256_setzero_pd();
        ymm5 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_2nx1m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_2N_1M(AlphaVal, b11, cs_b)
        /// implement TRSM///

        // extract a11
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack + 1));
        ymm5 = DTRSM_SMALL_DIV_OR_SCALE(ymm5, ymm0);

        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));

        //(Row 1): FMA operations
        ymm1 = _mm256_broadcast_sd((double const *)(a11 + cs_a));
        ymm3 = _mm256_fnmadd_pd(ymm1, ymm5, ymm3);

        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_2N_1M(b11, cs_b)

        m_remainder -= 1;
      }
    }
     n_remainder -= 2;
  }
  else if ( n_remainder == 1)
  {
    a01 = L + 1 * cs_a;     // pointer to block of A to be used in GEMM
    a11 = L; // pointer to block of A to be used for TRSM

    double *ptr_a10_dup = D_A_pack;

    dim_t p_lda = (n - 1); // packed leading dimension
    // perform copy of A to packed buffer D_A_pack

    if (transa)
    {
      for (dim_t x = 0; x < p_lda; x += 1)
      {
        bli_dcopys(*(a01), *(ptr_a10_dup));
        ptr_a10_dup += 1;
        a01 += cs_a;
      }
    }
    else
    {
      dim_t loop_count = (n - 1) / 4;
      for (dim_t x = 0; x < loop_count; x++)
      {
        ymm15 = _mm256_loadu_pd((double const *)(a01 + (rs_a * 0) + (x * 4)));
        _mm256_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (x * 4)), ymm15);
      }

      dim_t remainder_loop_count = p_lda - loop_count * 4;

      __m128d xmm0;
      if (remainder_loop_count != 0)
      {
        xmm0 = _mm_loadu_pd((double const *)(a01 + (rs_a * 0) + (loop_count * 4)));
        _mm_storeu_pd((double *)(ptr_a10_dup + (p_lda * 0) + (loop_count * 4)), xmm0);
      }
    }

    ymm4 = _mm256_broadcast_sd((double const *)&ones);
    if (!is_unitdiag)
    {
      // broadcast diagonal elements of A11
      ymm0 = _mm256_broadcast_sd((double const *)(a11));
      ymm1 = _mm256_broadcast_sd((double const *)&ones);
      ymm2 = _mm256_broadcast_sd((double const *)&ones);
      ymm3 = _mm256_broadcast_sd((double const *)&ones);

      ymm0 = _mm256_unpacklo_pd(ymm0, ymm1);
      ymm1 = _mm256_unpacklo_pd(ymm2, ymm3);

      ymm1 = _mm256_blend_pd(ymm0, ymm1, 0x0C);
#ifdef BLIS_DISABLE_TRSM_PREINVERSION
      ymm4 = ymm1;
#endif
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
      ymm4 = _mm256_div_pd(ymm4, ymm1);
#endif
    }
    _mm256_storeu_pd((double *)(d11_pack), ymm4);

    for (i = (m - d_mr); (i + 1) > 0; i -= d_mr) // loop along 'M' direction
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + i + 1*cs_b;                // pointer to block of B to be used in GEMM
      b11 = B + i;            // pointer to block of B to be used for TRSM

      k_iter = (n - 1); // number of GEMM operations to be done(in blocks of 4x4)

      /*Fill zeros into ymm registers used in gemm accumulations */
      ymm3 = _mm256_setzero_pd();
      ymm4 = _mm256_setzero_pd();
      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx8m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal);

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm1 = _mm256_loadu_pd((double const *)(b11 + 4));
      // B11[4][0] B11[5][0] B11[6][0] B11[7][0]

      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0
      ymm4 = _mm256_fmsub_pd(ymm1, ymm15, ymm4);
      // B11[4-7][0] * alpha-= ymm1

      /// implement TRSM///
      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);
      ymm4 = DTRSM_SMALL_DIV_OR_SCALE(ymm4, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);
      _mm256_storeu_pd((double *)(b11 + 4), ymm4);
    }

    dim_t m_remainder = i + d_mr;
    if (m_remainder >= 4)
    {
      a01 = D_A_pack;
      a11 = L; // pointer to block of A to be used for TRSM
      b10 = B + (m_remainder - 4) + 1*cs_b;        // pointer to block of B to be used in GEMM
      b11 = B + (m_remainder - 4);    // pointer to block of B to be used for TRSM

      k_iter = (n - 1); // number of GEMM operations to be done(in blocks of 4x4)

      ymm3 = _mm256_setzero_pd();

      /// GEMM implementation starts///
      BLIS_DTRSM_SMALL_GEMM_1nx4m(a01, b10, cs_b, p_lda, k_iter)

      ymm15 = _mm256_broadcast_sd((double const *)&AlphaVal); // register to hold alpha

      ymm0 = _mm256_loadu_pd((double const *)b11);
      // B11[0][0] B11[1][0] B11[2][0] B11[3][0]
      ymm3 = _mm256_fmsub_pd(ymm0, ymm15, ymm3);
      // B11[0-3][0] * alpha -= ymm0

      /// implement TRSM///
      // extract a00
      ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
      ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

      _mm256_storeu_pd((double *)b11, ymm3);

      m_remainder -= 4;
    }

    if (m_remainder)
    {
      if (m_remainder == 3)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 1*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;   // pointer to block of B to be used for TRSM

        k_iter = (n -  1); // number of GEMM operations to be done(in blocks of 4x4)

        ymm3 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_1nx3m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_1N_3M(AlphaVal, b11, cs_b)

        /// implement TRSM///
        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        ymm0 = _mm256_loadu_pd((double const *)b11);
        ymm3 = _mm256_blend_pd(ymm6, ymm3, 0x07);

        BLIS_POST_DTRSM_SMALL_1N_3M(b11, cs_b)

        m_remainder -= 3;
      }
      else if (m_remainder == 2)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 1*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n -  1); // number of GEMM operations to be done(in blocks of 4x4)

        ymm3 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_1nx2m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_1N_2M(AlphaVal, b11, cs_b)

        /// implement TRSM///
        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_1N_2M(b11, cs_b)

        m_remainder -= 2;
      }
      else if (m_remainder == 1)
      {
        a01 = D_A_pack;
        a11 = L; // pointer to block of A to be used for TRSM
        b10 = B + 1*cs_b;        // pointer to block of B to be used in GEMM
        b11 = B;    // pointer to block of B to be used for TRSM

        k_iter = (n - 1); // number of GEMM operations to be done(in blocks of 4x4)

        ymm3 = _mm256_setzero_pd();

        /// GEMM implementation starts///
        BLIS_DTRSM_SMALL_GEMM_1nx1m(a01, b10, cs_b, p_lda, k_iter)

        BLIS_PRE_DTRSM_SMALL_1N_1M(AlphaVal, b11, cs_b)

        /// implement TRSM///
        // extract a00
        ymm0 = _mm256_broadcast_sd((double const *)(d11_pack));
        ymm3 = DTRSM_SMALL_DIV_OR_SCALE(ymm3, ymm0);

        BLIS_POST_DTRSM_SMALL_1N_1M(b11, cs_b)

      }
    }
  }

  if ((required_packing_A) && bli_mem_is_alloc(&local_mem_buf_A_s))
  {
    bli_membrk_release(&rntm,
               &local_mem_buf_A_s);
  }
  return BLIS_SUCCESS;
}

// LLNN - LUTN
BLIS_INLINE err_t bli_dtrsm_small_AutXB_AlXB_AVX512
     (
       obj_t*   AlphaObj,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     )
{
  return BLIS_NOT_YET_IMPLEMENTED;
}

// LUNN LUTN
BLIS_INLINE err_t bli_dtrsm_small_AltXB_AuXB_AVX512
     (
       obj_t*   AlphaObj,
       obj_t*   a,
       obj_t*   b,
       cntx_t*  cntx,
       cntl_t*  cntl
     )
{
  return BLIS_NOT_YET_IMPLEMENTED;
}

#endif