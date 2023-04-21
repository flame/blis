/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

BLIS_EXPORT_BLAS void somatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const float* alpha, const float* aptr, f77_int* lda, float* bptr, f77_int* ldb);

BLIS_EXPORT_BLAS void domatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const double* alpha, const double* aptr, f77_int* lda, double* bptr, f77_int* ldb);

BLIS_EXPORT_BLAS void comatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const scomplex* alpha, const scomplex* aptr, f77_int* lda, scomplex* bptr, f77_int* ldb);

BLIS_EXPORT_BLAS void zomatcopy_ (f77_char* trans, f77_int* rows, f77_int* cols, const dcomplex* alpha, const dcomplex* aptr, f77_int* lda, dcomplex* bptr, f77_int* ldb);

#endif
