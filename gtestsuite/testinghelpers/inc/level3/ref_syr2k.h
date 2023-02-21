#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 *  SYR2K  performs one of the syr2ketric rank 2k operations
 *     C := alpha*A*B**T + alpha*B*A**T + beta*C,
 *  or
 *     C := alpha*A**T*B + alpha*B**T*A + beta*C,
 *  where  alpha and beta  are scalars, C is an  n by n  syr2ketric matrix
 *  and  A and B  are  n by k  matrices  in the  first  case  and  k by n
 *  matrices in the second case.
 *  ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syr2k(
    char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T* bp, gtint_t ldb,
    T beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers