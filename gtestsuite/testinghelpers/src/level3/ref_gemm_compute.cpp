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

#include "level3/ref_gemm_compute.h"

/*
 * ==========================================================================
 * GEMM Pack and Compute Extension performs the GEMM matrix-matrix operations
 * by first packing/reordering A/B matrix and computing the GEMM operation
 * on the packed buffer.
 *
 * Pack:
 *     Reorders the A or B matrix or both the matrices and scales them with
 *     alpha.
 *
 * Compute:
 *    C := A * B + beta*C,
 * where,
 *      Either A or B or both A and B matrices are packed matrices.
 *      Alpha and beta are scalars, and A, B and C are matrices, with A
 *      an m by k matrix, B a k by n matrix and C an m by n matrix,
 *      where either A or B or both may be scaled by alpha and reordered.
 *
 * NOTE:
 * - For MKL comparing against pack and compute APIs.
 * - For all other reference libraries (except MKL), we compare the result of
 *   BLIS pack and compute against the GEMM operation of the reference library.
 *   In case when both A & B are unpacked, we do not invoke xgemm_pack() thus,
 *   not computing alpha * X operation. So to handle this case, we pass
 *   unit-alpha to the reference GEMM.
 * ==========================================================================
 */

namespace testinghelpers {

#ifdef REF_IS_MKL
template <typename T>
void ref_gemm_compute(char storage, char trnsa, char trnsb, char pcka, char pckb, gtint_t m, gtint_t n, gtint_t k, T alpha,
    T* ap, gtint_t lda, T* bp, gtint_t ldb, T beta, T* cp, gtint_t ldc)
{
    T unit_alpha = 1.0;
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trnsa, &cblas_transa );
    char_to_cblas_trans( trnsb, &cblas_transb );

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;

    typedef gint_t (*Fptr_ref_cblas_gemm_pack_get_size)( const CBLAS_IDENTIFIER,
                    const f77_int, const f77_int, const f77_int );
    Fptr_ref_cblas_gemm_pack_get_size ref_cblas_gemm_pack_get_size;

    typedef void (*Fptr_ref_cblas_gemm_pack)( const CBLAS_ORDER, const CBLAS_IDENTIFIER, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const f77_int, const T, const T*, f77_int,
                    T*);
    Fptr_ref_cblas_gemm_pack ref_cblas_gemm_pack;

    typedef void (*Fptr_ref_cblas_gemm_compute)( const CBLAS_ORDER, const f77_int, const f77_int,
                    const f77_int, const f77_int, const f77_int, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemm_compute ref_cblas_gemm_compute;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemm_pack_get_size = (Fptr_ref_cblas_gemm_pack_get_size)refCBLASModule.loadSymbol("cblas_sgemm_pack_get_size");
        ref_cblas_gemm_pack          = (Fptr_ref_cblas_gemm_pack)refCBLASModule.loadSymbol("cblas_sgemm_pack");
        ref_cblas_gemm_compute       = (Fptr_ref_cblas_gemm_compute)refCBLASModule.loadSymbol("cblas_sgemm_compute");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemm_pack_get_size = (Fptr_ref_cblas_gemm_pack_get_size)refCBLASModule.loadSymbol("cblas_dgemm_pack_get_size");
        ref_cblas_gemm_pack          = (Fptr_ref_cblas_gemm_pack)refCBLASModule.loadSymbol("cblas_dgemm_pack");
        ref_cblas_gemm_compute       = (Fptr_ref_cblas_gemm_compute)refCBLASModule.loadSymbol("cblas_dgemm_compute");
    }
    else
    {
        throw std::runtime_error("Error in ref_gemm_compute.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_gemm_compute ) {
        throw std::runtime_error("Error in ref_gemm_compute.cpp: Function pointer == 0 -- symbol not found.");
    }

    err_t err = BLIS_SUCCESS;

    if ( ( pcka == 'P' || pcka == 'p' ) && ( pckb == 'P' || pckb == 'p' ) )
    {
        // Reorder A
        CBLAS_IDENTIFIER cblas_identifierA = CblasAMatrix;
        CBLAS_STORAGE cblas_packed = CblasPacked;
        gtint_t bufSizeA = ref_cblas_gemm_pack_get_size( cblas_identifierA,
                                                         m,
                                                         n,
                                                         k );

        T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

        ref_cblas_gemm_pack( cblas_order, cblas_identifierA, cblas_transa,
                m, n, k, alpha, ap, lda, aBuffer );

        // Reorder B
        CBLAS_IDENTIFIER cblas_identifierB = CblasBMatrix;
        gtint_t bufSizeB = ref_cblas_gemm_pack_get_size( cblas_identifierB,
                                                         m,
                                                         n,
                                                         k );

        T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

        ref_cblas_gemm_pack( cblas_order, cblas_identifierB, cblas_transb,
                m, n, k, unit_alpha, bp, ldb, bBuffer );

        ref_cblas_gemm_compute( cblas_order, cblas_packed, cblas_packed,
                m, n, k, aBuffer, lda, bBuffer, ldb, beta, cp, ldc );

        bli_free_user( aBuffer );
        bli_free_user( bBuffer );
    }
    else if ( ( pcka == 'P' || pcka == 'p' ) )
    {
        // Reorder A
        CBLAS_IDENTIFIER cblas_identifier = CblasAMatrix;
        CBLAS_STORAGE cblas_packed = CblasPacked;
        gtint_t bufSizeA = ref_cblas_gemm_pack_get_size( cblas_identifier,
                                                         m,
                                                         n,
                                                         k );

        T* aBuffer = (T*) bli_malloc_user( bufSizeA, &err );

        ref_cblas_gemm_pack( cblas_order, cblas_identifier, cblas_transa,
                m, n, k, alpha, ap, lda, aBuffer );

        ref_cblas_gemm_compute( cblas_order, cblas_packed, cblas_transb,
                m, n, k, aBuffer, lda, bp, ldb, beta, cp, ldc );

        bli_free_user( aBuffer );
    }
    else if ( ( pckb == 'P' || pckb == 'p' ) )
    {
        // Reorder B
        CBLAS_IDENTIFIER cblas_identifier = CblasBMatrix;
        CBLAS_STORAGE cblas_packed = CblasPacked;
        gtint_t bufSizeB = ref_cblas_gemm_pack_get_size( cblas_identifier,
                                                         m,
                                                         n,
                                                         k );

        T* bBuffer = (T*) bli_malloc_user( bufSizeB, &err );

        ref_cblas_gemm_pack( cblas_order, cblas_identifier, cblas_transb,
                m, n, k, alpha, bp, ldb, bBuffer );

        ref_cblas_gemm_compute( cblas_order, cblas_transa, cblas_packed,
                m, n, k, ap, lda, bBuffer, ldb, beta, cp, ldc );

        bli_free_user( bBuffer );
    }
    else
    {
        ref_cblas_gemm_compute( cblas_order, cblas_transa, cblas_transb,
                m, n, k, ap, lda, bp, ldb, beta, cp, ldc );
    }
}
#else
template <typename T>
void ref_gemm_compute(char storage, char trnsa, char trnsb, char pcka, char pckb, gtint_t m, gtint_t n, gtint_t k, T alpha,
    T* ap, gtint_t lda, T* bp, gtint_t ldb, T beta, T* cp, gtint_t ldc)
{
    // throw std::runtime_error("Error in ref_gemm_compute.cpp: Reference is only defined for MKL. Please use MKL as reference library.");
    enum CBLAS_ORDER cblas_order;
    enum CBLAS_TRANSPOSE cblas_transa;
    enum CBLAS_TRANSPOSE cblas_transb;

    char_to_cblas_order( storage, &cblas_order );
    char_to_cblas_trans( trnsa, &cblas_transa );
    char_to_cblas_trans( trnsb, &cblas_transb );

    using scalar_t = std::conditional_t<testinghelpers::type_info<T>::is_complex, T&, T>;
    typedef void (*Fptr_ref_cblas_gemm)( const CBLAS_ORDER, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
                    const f77_int, const f77_int, const f77_int, const scalar_t, const T*, f77_int,
                    const T*, f77_int, const scalar_t, T*, f77_int);
    Fptr_ref_cblas_gemm ref_cblas_gemm;

    // Call C function
    /* Check the typename T passed to this function template and call respective function.*/
    if (typeid(T) == typeid(float))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)refCBLASModule.loadSymbol("cblas_sgemm");
    }
    else if (typeid(T) == typeid(double))
    {
        ref_cblas_gemm = (Fptr_ref_cblas_gemm)refCBLASModule.loadSymbol("cblas_dgemm");
    }
    else
    {
        throw std::runtime_error("Error in ref_gemm.cpp: Invalid typename is passed function template.");
    }
    if( !ref_cblas_gemm ) {
        throw std::runtime_error("Error in ref_gemm.cpp: Function pointer == 0 -- symbol not found.");
    }

    if ( ( pcka == 'U' or pcka == 'u' ) && ( pckb == 'U' or pckb == 'u' ) )
    {
        T unit_alpha = 1.0;
        ref_cblas_gemm( cblas_order, cblas_transa, cblas_transb,
                        m, n, k, unit_alpha, ap, lda, bp, ldb, beta, cp, ldc );
    }
    else
    {
        ref_cblas_gemm( cblas_order, cblas_transa, cblas_transb,
                        m, n, k, alpha, ap, lda, bp, ldb, beta, cp, ldc );
    }
}
#endif

// Explicit template instantiations
template void ref_gemm_compute<float>(char, char, char, char, char, gtint_t, gtint_t, gtint_t, float,
                      float*, gtint_t, float*, gtint_t, float, float*, gtint_t );
template void ref_gemm_compute<double>(char, char, char, char, char, gtint_t, gtint_t, gtint_t, double,
                      double*, gtint_t, double*, gtint_t, double, double*, gtint_t );

} //end of namespace testinghelpers
