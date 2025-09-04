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

#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  HER2K  performs one of the symmetric rank 2k operations
 *     C := alpha*A*B**H + alpha*B*A**H + beta*C,
 *  or
 *    C := alpha*A**T*B + alpha*B**T*A + beta*C,
 *  where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
 *  and  A and B  are  n by k  matrices  in the  first  case  and  k by n
 *  matrices in the second case.
 *  ==========================================================================
**/

namespace testinghelpers {

template <typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
void ref_her2k(
    char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k,
    T* alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    RT beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers
