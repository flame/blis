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
#ifdef BLIS_ENABLE_CBLAS

#include "cblas.h"
#include "cblas_f77.h"

BLIS_EXPORT_BLAS void cblas_dgemm_compute( enum  CBLAS_ORDER Order,
                                                 f77_int TransA,
                                                 f77_int TransB,
                                           const f77_int M, const f77_int N,
                                           const f77_int K,
                                           const double* A,       f77_int lda,
                                           const double* B,       f77_int ldb,
                                                 double  beta,
                                                 double* C,       f77_int ldc )
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

    if ( Order == CblasColMajor )       // CblasColMajor
    {
        if      ( TransA == CblasTrans )     TA='T';
        else if ( TransA == CblasConjTrans ) TA='T';
        else if ( TransA == CblasNoTrans )   TA='N';
        else if ( TransA == CblasPacked )    TA='P';
        else
        {
            cblas_xerbla(2, "cblas_dgemm_compute",
                            "Illegal TransA setting, %d\n", TransA);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

        if      ( TransB == CblasTrans )     TB='T';
        else if ( TransB == CblasConjTrans ) TB='T';
        else if ( TransB == CblasNoTrans )   TB='N';
        else if ( TransB == CblasPacked )    TB='P';
        else
        {
            cblas_xerbla(3, "cblas_dgemm_compute",
                            "Illegal TransB setting, %d\n", TransB);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
#endif

        f77_int rs_a = 1;
        f77_int rs_b = 1;
        f77_int rs_c = 1;

        F77_dgemm_compute( F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, A, &rs_a, &F77_lda,
                           B, &rs_b, &F77_ldb, &beta, C, &rs_c, &F77_ldc);
    }
    else if ( Order == CblasRowMajor )      // CblasRowMajor
    {
        RowMajorStrg = 1;

        // If Row Major, and A is not already reordered
        // then toggle the transA parameter and interchange the strides.
        if      ( TransA == CblasPacked )    TA='P';
        else if ( TransA == CblasTrans )     TA='N';
        else if ( TransA == CblasNoTrans )   TA='T';
        else if ( TransA == CblasConjTrans ) TA='N';
        else
        {
            cblas_xerbla(2, "cblas_dgemm_compute",
                            "Illegal TransA setting, %d\n", TransA);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

        // If Row Major, and B is not already reordered
        // then toggle the transB parameter and interchange the strides.
        if      ( TransB == CblasPacked )    TB='P';
        else if ( TransB == CblasTrans )     TB='N';
        else if ( TransB == CblasNoTrans )   TB='T';
        else if ( TransB == CblasConjTrans ) TB='N';
        else
        {
            cblas_xerbla(2, "cblas_dgemm_compute",
                            "Illegal TransB setting, %d\n", TransB);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

#ifdef F77_CHAR
        F77_TA = C2F_CHAR(&TA);
        F77_TB = C2F_CHAR(&TB);
#endif

        f77_int rs_a = 1;
        f77_int rs_b = 1;
        f77_int cs_c = 1;

        F77_dgemm_compute( F77_TA, F77_TB, &F77_M, &F77_N, &F77_K, A, &rs_a, &F77_lda,
                           B, &rs_b, &F77_ldb, &beta, C, &F77_ldc, &cs_c );
    }
    else
    {
        cblas_xerbla(1, "cblas_dgemm_compute",
                        "Illegal Order setting, %d\n", Order);
        CBLAS_CallFromC = 0;
        RowMajorStrg = 0;
        return;
    }
    return;
}
#endif
