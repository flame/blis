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

#ifdef BLIS_ENABLE_TRSM_PREINVERSION
  #define DIAG_ELE_INV_OPS(a, b) (a / b)
  #define DIAG_ELE_EVAL_OPS(a, b) (a * b)
#endif

#ifdef BLIS_DISABLE_TRSM_PREINVERSION
  #define DIAG_ELE_INV_OPS(a, b) (a * b)
  #define DIAG_ELE_EVAL_OPS(a, b) (a / b)
#endif

// reference code for LUTN
BLIS_INLINE err_t dtrsm_AutXB_ref
   (
      double *A,
      double *B,
      dim_t M,
      dim_t N,
      dim_t lda,
      dim_t ldb,
      bool unitDiagonal
   )
{
  dim_t i, j, k;
  for (k = 0; k < M; k++)
  {
    double lkk_inv = 1.0;
    if (!unitDiagonal)
      lkk_inv = DIAG_ELE_INV_OPS(lkk_inv, A[k + k * lda]);
    for (j = 0; j < N; j++)
    {
      B[k + j * ldb] = DIAG_ELE_EVAL_OPS(B[k + j * ldb], lkk_inv);
      for (i = k + 1; i < M; i++)
      {
        B[i + j * ldb] -= A[i * lda + k] * B[k + j * ldb];
      }
    }
  } // k -loop
  return BLIS_SUCCESS;
}

// reference code for LLNN
BLIS_INLINE err_t dtrsm_AlXB_ref
   (
      double *A,
      double *B,
      dim_t M,
      dim_t N,
      dim_t lda,
      dim_t ldb,
      bool is_unitdiag
    )
{
  dim_t i, j, k;
  for (k = 0; k < M; k++)
  {
    double lkk_inv = 1.0;
    if (!is_unitdiag)
      lkk_inv = DIAG_ELE_INV_OPS(lkk_inv, A[k + k * lda]);
    for (j = 0; j < N; j++)
    {
      B[k + j * ldb] = DIAG_ELE_EVAL_OPS(B[k + j * ldb], lkk_inv);
      for (i = k + 1; i < M; i++)
      {
        B[i + j * ldb] -= A[i + k * lda] * B[k + j * ldb];
      }
    }
  } // k -loop
  return BLIS_SUCCESS;
}

// reference code for LUNN
BLIS_INLINE err_t dtrsm_AuXB_ref
   (
     double *A,
     double *B,
     dim_t M,
     dim_t N,
     dim_t lda,
     dim_t ldb,
     bool is_unitdiag
   )
{
  dim_t i, j, k;
  for (k = M - 1; k >= 0; k--)
  {
    double lkk_inv = 1.0;
    if (!is_unitdiag)
      lkk_inv = DIAG_ELE_INV_OPS(lkk_inv, A[k + k * lda]);
    for (j = N - 1; j >= 0; j--)
    {
      B[k + j * ldb] = DIAG_ELE_EVAL_OPS(B[k + j * ldb], lkk_inv);
      for (i = k - 1; i >= 0; i--)
      {
        B[i + j * ldb] -= A[i + k * lda] * B[k + j * ldb];
      }
    }
  } // k -loop
  return BLIS_SUCCESS;
} // end of function

// reference code for LLTN
BLIS_INLINE err_t dtrsm_AltXB_ref
   (
     double *A,
     double *B,
     dim_t M,
     dim_t N,
     dim_t lda,
     dim_t ldb,
     bool is_unitdiag
   )
{
  dim_t i, j, k;
  for (k = M - 1; k >= 0; k--)
  {
    double lkk_inv = 1.0;
    if (!is_unitdiag)
      lkk_inv = DIAG_ELE_INV_OPS(lkk_inv, A[k + k * lda]);
    for (j = N - 1; j >= 0; j--)
    {
      B[k + j * ldb] = DIAG_ELE_EVAL_OPS(B[k + j * ldb], lkk_inv);
      for (i = k - 1; i >= 0; i--)
      {
        B[i + j * ldb] -= A[i * lda + k] * B[k + j * ldb];
      }
    }
  } // k -loop
  return BLIS_SUCCESS;
} // end of function
