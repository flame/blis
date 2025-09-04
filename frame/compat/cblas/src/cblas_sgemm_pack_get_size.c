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

f77_int cblas_sgemm_pack_get_size( enum  CBLAS_IDENTIFIER Identifier,
                                   const f77_int M,
                                   const f77_int N,
                                   const f77_int K )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_1 );

    char ID;
    f77_int tbytes = 0;

#ifdef F77_CHAR
    F77_CHAR F77_ID;
#else
    #define F77_ID &ID
#endif

#ifdef F77_INT
    F77_INT F77_M=M, F77_N=N, F77_K=K;
#else
    #define F77_M M
    #define F77_N N
    #define F77_K K
#endif

    if      ( Identifier == CblasAMatrix ) ID = 'A';
    else if ( Identifier == CblasBMatrix ) ID = 'B';
    else
     {
        cblas_xerbla( 1, "cblas_sgemm_pack_get_size",
                         "Illegal CBLAS_IDENTIFIER setting, %d\n", Identifier );
        AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_1 );
        return 0;
     }

#ifdef F77_CHAR
    F77_ID = C2F_CHAR( &ID );
#endif
    tbytes = F77_sgemm_pack_get_size ( F77_ID, &F77_M, &F77_N, &F77_K );

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_1 );
    return tbytes;
}
#endif
