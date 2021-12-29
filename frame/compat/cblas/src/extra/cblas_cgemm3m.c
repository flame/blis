#include "blis.h"
#ifdef BLIS_ENABLE_CBLAS
/*
 *
 * cblas_cgemm3m.c
 *
 * This program is a C interface to cgemm3m.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 *
 */

#include "cblas.h"
#include "cblas_f77.h"
void cblas_cgemm3m(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, f77_int M, f77_int N,
                 f77_int K, const void *alpha, const void  *A,
                 f77_int lda, const void  *B, f77_int ldb,
                 const void *beta, void  *C, f77_int ldc)
{
   char TA, TB;   
#ifdef F77_CHAR
   F77_CHAR F77_TA, F77_TB;
#else
   #define F77_TA &TA  
   #define F77_TB &TB  
#endif

#ifdef F77_INT
   F77_INT F77_M=M, F77_N=N, F77_K=K, F77_lda=lda, F77_ldb=ldb;
   F77_INT F77_ldc=ldc;
#else
   #define F77_M M
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
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm3m", "Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else 
      {
         cblas_xerbla(3, "cblas_cgemm3m", "Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }

      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif

      F77_cgemm3m(F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, (scomplex*)alpha, (scomplex*)A,
                     &F77_lda, (scomplex*)B, &F77_ldb, (scomplex*)beta, (scomplex*)C, &F77_ldc);
   } else if (Order == CblasRowMajor)
   {
      RowMajorStrg = 1;
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm3m", "Illegal TransA setting, %d\n", TransA);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else 
      {
         cblas_xerbla(2, "cblas_cgemm3m", "Illegal TransB setting, %d\n", TransB);
         CBLAS_CallFromC = 0;
         RowMajorStrg = 0;
         return;
      }
      #ifdef F77_CHAR
         F77_TA = C2F_CHAR(&TA);
         F77_TB = C2F_CHAR(&TB);
      #endif
      

      F77_cgemm3m(F77_TA, F77_TB, &F77_N, &F77_M, &F77_K, (scomplex*)alpha, (scomplex*)B,
                  &F77_ldb, (scomplex*)A, &F77_lda, (scomplex*)beta, (scomplex*)C, &F77_ldc);
   } 
   else cblas_xerbla(1, "cblas_cgemm3m", "Illegal Order setting, %d\n", Order);
   CBLAS_CallFromC = 0;
   RowMajorStrg = 0;
   return;
}
#endif
