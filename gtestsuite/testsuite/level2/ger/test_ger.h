#pragma once

#include "ger.h"
#include "level2/ref_ger.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>

void test_ger( char storage, char conjx, char conjy, gtint_t m, gtint_t n,
    T alpha, gtint_t incx, gtint_t incy, gtint_t lda_inc, double thresh,
    char datatype ) {

    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', m, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 5, storage, 'n', m, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(-3, 3, m, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(-3, 3, n, incy, datatype);

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    ger( storage, conjx, conjy, m, n, &alpha, x.data(), incx,
                                              y.data(), incy, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_ger( storage, conjx, conjy, m, n, alpha,
                          x.data(), incx, y.data(), incy, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, a.data(), a_ref.data(), lda, thresh );
}