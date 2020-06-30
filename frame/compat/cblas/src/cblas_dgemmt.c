#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_dgemmt.c
 * This program is a C interface to dgemmt.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc.
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_dgemmt( enum CBLAS_ORDER Order, enum CBLAS_UPLO Uplo,
		   enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB,
		   f77_int N, f77_int K,
		   double alpha, const double  *A,
                   f77_int lda, const double  *B, f77_int ldb,
                   double beta, double  *C, f77_int ldc)
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
   char TA, TB, UL;
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_TB, F77_UL;
#else
   #define F77_TA &TA
   #define F77_TB &TB
   #define F77_UL &UL
#endif

#ifdef F77_INT
   F77_INT F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_N N
   #define F77_K K
   #define F77_lda lda
   #define F77_ldb ldb
   #define F77_ldc ldc
#endif

   extern int CBLAS_CallFromC;
   extern int RowMajorStrg;
   RowMajorStrg = 0;
   CBLAS_CallFromC = 1;

   if( Order == CblasColMajor )
   {
      if( Uplo == CblasUpper) UL = 'U';
      else if(Uplo == CblasLower) UL = 'L';
      else
      {
         cblas_xerbla(2, "cblas_dgemmt","Illegal Uplo setting, %d\n", Uplo);
	 CBLAS_CallFromC = 0;
	 RowMajorStrg = 0;
	 return;
      }

      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(3, "cblas_dgemmt","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(4, "cblas_dgemmt","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
	 F77_UL = C2F_CHAR(&UL);
      #endif

      F77_dgemmt(F77_UL,F77_TA, F77_TB, &F77_N, &F77_K, &alpha, A,
       &F77_lda, B, &F77_ldb, &beta, C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if(Uplo == CblasUpper) UL = 'L';
      else if(Uplo == CblasLower) UL = 'U';
      else
      {
         cblas_xerbla(2, "cblas_dgemmt","Illegal Uplo setting, %d\n", Uplo);
      }

      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(3, "cblas_dgemmt","Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(4, "cblas_dgemmt","Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
	 F77_UL = C2F_CHAR(&UL);
      #endif

      F77_dgemmt(F77_UL,F77_TA, F77_TB, &F77_N, &F77_K, &alpha, B,
                  &F77_ldb, A, &F77_lda, &beta, C, &F77_ldc);
	  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
   }
   else  cblas_xerbla(1, "cblas_dgemmt", "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
   return;
}
#endif
