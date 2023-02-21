#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  C := alpha*A*A**H + beta*C,
 *       or
 *  C := alpha*A**H*A + beta*C,
 *  ==========================================================================
**/

namespace testinghelpers {

template <typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
void ref_herk(
    char storage, char uplo, char transa,
    gtint_t m, gtint_t k,
    RT alpha,
    T* ap, gtint_t lda,
    RT beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers