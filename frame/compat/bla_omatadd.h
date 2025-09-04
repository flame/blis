/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

//
// Prototype BLAS-to-BLIS interfaces.
//

#ifdef BLIS_ENABLE_BLAS

BLIS_EXPORT_BLAS void somatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc);

BLIS_EXPORT_BLAS void domatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc);

BLIS_EXPORT_BLAS void comatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc);

BLIS_EXPORT_BLAS void zomatadd_ (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc);

#endif

BLIS_EXPORT_BLAS void somatadd_blis_impl (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const float* alpha, const float* A, f77_int* lda, const float* beta, const float* B, f77_int* ldb, float* C, f77_int* ldc);

BLIS_EXPORT_BLAS void domatadd_blis_impl (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const double* alpha, const double* A, f77_int* lda, const double* beta, const double* B, f77_int* ldb, double* C, f77_int* ldc);

BLIS_EXPORT_BLAS void comatadd_blis_impl (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const scomplex* alpha, const scomplex* A, f77_int* lda,const scomplex* beta, scomplex* B, f77_int* ldb, scomplex* C, f77_int* ldc);

BLIS_EXPORT_BLAS void zomatadd_blis_impl (f77_char* transa,f77_char* transb, f77_int* m, f77_int* n, const dcomplex* alpha, const dcomplex* A, f77_int* lda,const dcomplex* beta, dcomplex* B, f77_int* ldb, dcomplex* C, f77_int* ldc);
