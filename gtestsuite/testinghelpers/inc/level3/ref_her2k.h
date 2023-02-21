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