#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_dtrsm.c
 * This program is a C interface to dtrsm.
 * Written by Keita Teranishi
 * 4/6/1998
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dtrsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side,
                 enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, f77_int M, f77_int N,
                 double alpha, const double  *A, f77_int lda,
                 double  *B, f77_int ldb)

{
   AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
   char UL, TA, SD, DI;   
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_UL, F77_SD, F77_DI;
#else
   #define F77_TA &TA  
   #define F77_UL &UL  
   #define F77_SD &SD
   #define F77_DI &DI
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_lda=lda, F77_ldb=ldb;
#else
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
   #define F77_ldb ldb
#endif

   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;
   CBLAS_CallFromC = 1;

   if( Order == CblasColMajor )
   {
      if      ( Side == CblasRight) SD='R';
      else if ( Side == CblasLeft ) SD='L';
      else 
      {
         cblas_xerbla(2, "cblas_dtrsm","Illegal Side setting, %d\n", Side);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal side setting.");
         return;
      }
      if      ( Uplo == CblasUpper) UL='U';
      else if ( Uplo == CblasLower) UL='L';
      else 
      {
         cblas_xerbla(3, "cblas_dtrsm","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal uplo setting.");
         return;
      }

      if      ( TransA == CblasTrans    ) TA='T';
      else if ( TransA == CblasConjTrans) TA='C';
      else if ( TransA == CblasNoTrans  ) TA='N';
      else 
      {
         cblas_xerbla(4, "cblas_dtrsm","Illegal Trans setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal trans setting.");
         return;
      }

      if      ( Diag == CblasUnit   ) DI='U';
      else if ( Diag == CblasNonUnit) DI='N';
      else 
      {
         cblas_xerbla(5, "cblas_dtrsm","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal diag setting.");
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_SD = C2F_CHAR(&SD);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_dtrsm(F77_SD, F77_UL, F77_TA, F77_DI, &F77_M, &F77_N, &alpha,
                A, &F77_lda, B, &F77_ldb);
   } 
   else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if      ( Side == CblasRight) SD='L';
      else if ( Side == CblasLeft ) SD='R';
      else 
      {
         cblas_xerbla(2, "cblas_dtrsm","Illegal Side setting, %d\n", Side);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal side setting.");
         return;
      }

      if      ( Uplo == CblasUpper) UL='L';
      else if ( Uplo == CblasLower) UL='U';
      else 
      {
         cblas_xerbla(3, "cblas_dtrsm","Illegal Uplo setting, %d\n", Uplo);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal uplo setting.");
         return;
      }

      if      ( TransA == CblasTrans    ) TA='T';
      else if ( TransA == CblasConjTrans) TA='C';
      else if ( TransA == CblasNoTrans  ) TA='N';
      else 
      {
         cblas_xerbla(4, "cblas_dtrsm","Illegal Trans setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal trans setting.");
         return;
      }

      if      ( Diag == CblasUnit   ) DI='U';
      else if ( Diag == CblasNonUnit) DI='N';
      else 
      {
         cblas_xerbla(5, "cblas_dtrsm","Illegal Diag setting, %d\n", Diag);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal diag setting.");
         return;
      }

      #ifdef F77_CHAR
         F77_UL = C2F_CHAR(&UL);
         F77_TA = C2F_CHAR(&TA);
         F77_SD = C2F_CHAR(&SD);
         F77_DI = C2F_CHAR(&DI);
      #endif

      F77_dtrsm(F77_SD, F77_UL, F77_TA, F77_DI, &F77_N, &F77_M, &alpha, A, 
               &F77_lda, B, &F77_ldb);
   } 
   else
   {
      cblas_xerbla(1, "cblas_dtrsm","Illegal Order setting, %d\n", Order);
      CBLAS_CallFromC = 0;
      RowMajorStrg = 0;
      AOCL_DTL_TRACE_EXIT_ERR(AOCL_DTL_LEVEL_TRACE_1, "Illegal order setting.");
      return;
   }
   AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
   return;
}
#endif
