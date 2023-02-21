#pragma once

#include "her2k.h"
#include "level3/ref_her2k.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T, typename RT = typename testinghelpers::type_info<T>::real_type>
void test_her2k( char storage, char uplo, char transa, char transb,
    gtint_t m, gtint_t k,
    gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
    T alpha, RT beta,
    double thresh, char datatype
) {
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, m, k, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, transb, m, k, ldb_inc);
    gtint_t ldc = testinghelpers::get_leading_dimension(storage, 'n', m, m, ldc_inc);

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 8, storage, transa, m, k, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-5, 2, storage, transb, m, k, ldb, datatype);
    // Since matrix C, stored in c, is symmetric and we only use the upper or lower
    // part in the computation of her2k and zero-out the rest to ensure
    // that code operates as expected.
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-3, 5, storage, uplo, m, ldc, datatype);

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    her2k<T>( storage, uplo, transa, transb, m, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_her2k( storage, uplo, transa, transb, m, k, &alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, m, c.data(), c_ref.data(), ldc, thresh );
}
