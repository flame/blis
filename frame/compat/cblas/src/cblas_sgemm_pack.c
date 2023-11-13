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

BLIS_EXPORT_BLAS void cblas_sgemm_pack( enum  CBLAS_ORDER      Order,
                                        enum  CBLAS_IDENTIFIER Identifier,
                                        enum  CBLAS_TRANSPOSE  Trans,
                                        const f77_int M,
                                        const f77_int N,
                                        const f77_int K,
                                        const float   alpha,
                                        const float*  src, const f77_int ld,
                                              float*  dest )
{
    char TR;
    char ID;

#ifdef F77_CHAR
    F77_CHAR F77_TR;
    F77_CHAR F77_ID;
#else
#define F77_TR &TR
#define F77_ID &ID
#endif

#ifdef F77_INT
    F77_INT F77_M=M, F77_N=N, F77_K=K, F77_ld=ld;
#else

#define F77_M M
#define F77_N N
#define F77_K K
#define F77_ld ld

#endif

    extern int CBLAS_CallFromC;
    extern int RowMajorStrg;
    RowMajorStrg = 0;

    CBLAS_CallFromC = 1;

    if ( Order == CblasColMajor )       // CblasColMajor
    {
        if      ( Trans == CblasNoTrans )   TR = 'N';
        else if ( Trans == CblasTrans )     TR = 'T';
        else if ( Trans == CblasConjTrans ) TR = 'T';
        else
        {
            cblas_xerbla(3, "cblas_sgemm_pack","Illegal Trans setting, %d\n", Trans);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

        if      ( Identifier == CblasAMatrix ) ID = 'A';
        else if ( Identifier == CblasBMatrix ) ID = 'B';
        else
        {
            cblas_xerbla(3, "cblas_sgemm_pack","Illegal Identifier setting, %d\n", Identifier);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

#ifdef F77_CHAR
        F77_TR = C2F_CHAR(&TR);
        F77_ID = C2F_CHAR(&ID);
#endif
        F77_sgemm_pack( F77_ID,
                        F77_TR,
                        &F77_M,
                        &F77_N,
                        &F77_K,
                        &alpha,
                        src, &F77_ld,
                        dest );
    }
    else if ( Order == CblasRowMajor )      // CblasRowMajor
    {
        RowMajorStrg = 1;
        if      ( Trans == CblasNoTrans )   TR = 'T';
        else if ( Trans == CblasTrans )     TR = 'N';
        else if ( Trans == CblasConjTrans ) TR = 'N';
        else
        {
            cblas_xerbla(3, "cblas_sgemm_pack","Invalid Trans setting, %d\n", Trans);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

        if      ( Identifier == CblasAMatrix ) ID = 'A';
        else if ( Identifier == CblasBMatrix ) ID = 'B';
        else
        {
            cblas_xerbla(3, "cblas_sgemm_pack","Illegal Identifier setting, %d\n", Identifier);
            CBLAS_CallFromC = 0;
            RowMajorStrg = 0;
            return;
        }

#ifdef F77_CHAR
        F77_TR = C2F_CHAR(&TR);
        F77_ID = C2F_CHAR(&ID);
#endif
        F77_sgemm_pack ( F77_ID,
                         F77_TR,
                         &F77_M,
                         &F77_N,
                         &F77_K,
                         &alpha,
                         src, &F77_ld,
                         dest );
    }
    else cblas_xerbla(1, "cblas_sgemm_pack", "Invalid Order setting, %d\n", Order);
    CBLAS_CallFromC = 0;
    RowMajorStrg = 0;
    return;
}
#endif
