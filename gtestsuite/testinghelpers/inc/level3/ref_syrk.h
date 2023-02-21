#pragma once

#include "common/testing_helpers.h"

/*
 *  ==========================================================================
 *  C := alpha*A*A**T + beta*C,
 *       or
 *  C := alpha*A**T*A + beta*C,
 *  ==========================================================================
**/

namespace testinghelpers {

template <typename T>
void ref_syrk(
    char storage, char uplo, char transa,
    gtint_t m, gtint_t k,
    T alpha,
    T* ap, gtint_t lda,
    T beta,
    T* cp, gtint_t ldc
);

} //end of namespace testinghelpers