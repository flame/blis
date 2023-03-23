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
