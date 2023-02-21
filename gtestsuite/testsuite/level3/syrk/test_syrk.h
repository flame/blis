#pragma once

#include "syrk.h"
#include "level3/ref_syrk.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_syrk( char storage, char uplo, char transa,
    gtint_t m, gtint_t k,
    gtint_t lda_inc, gtint_t ldc_inc,
    T alpha, T beta,
    double thresh, char datatype
) {
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, m, k, lda_inc);
    gtint_t ldc = testinghelpers::get_leading_dimension(storage, 'n', m, m, ldc_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>( -2, 8, storage, transa, m, k, lda, datatype );
    // Since matrix C, stored in c, is symmetric, we only use the upper or lower
    // part in the computation of syrk and zero-out the rest to ensure
    // that code operates as expected.
    std::vector<T> c = testinghelpers::get_random_matrix<T>( -3, 5, storage, uplo, m, ldc, datatype );

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    syrk<T>( storage, uplo, transa, m, k, &alpha, a.data(), lda,
                &beta, c.data(), ldc );
    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_syrk<T>( storage, uplo, transa, m, k, alpha,
               a.data(), lda, beta, c_ref.data(), ldc );
    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, m, c.data(), c_ref.data(), ldc, thresh );
}
