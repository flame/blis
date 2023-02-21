#pragma once

#include "hemm.h"
#include "level3/ref_hemm.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_hemm( char storage, char side, char uplo, char conja, char transb,
    gtint_t m, gtint_t n,
    gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
    T alpha, T beta,
    double thresh, char datatype
) {
    // Set the dimension for row/col of A, depending on the value of side.
    gtint_t k = ((side == 'l')||(side == 'L'))? m : n;
    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', k, k, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldb_inc);
    gtint_t ldc = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldc_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    // Since matrix A, stored in a, is symmetric and we only use the upper or lower
    // part in the computation of hemm and zero-out the rest to ensure
    // that code operates as expected.
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-5, 2, storage, uplo, k, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-5, 2, storage, transb, m, n, ldb, datatype);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-3, 5, storage, 'n', m, n, ldc, datatype);
    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    hemm<T>( storage, side, uplo, conja, transb, m, n, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_hemm( storage, side, uplo, conja, transb, m, n, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldc, thresh );
}
