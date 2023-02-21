#pragma once

#include "syr2.h"
#include "level2/ref_syr2.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_syr2( char storage, char uploa, char conjx, char conjy, gtint_t n,
    T alpha, gtint_t incx, gtint_t incy, gtint_t lda_inc, double thresh,
    char datatype ) {

    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', n, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 5, storage, 'n', n, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(-3, 3, n, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(-3, 3, n, incy, datatype);

    mksymm<T>( storage, uploa, n, a.data(), lda );
    mktrim<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    syr2<T>( storage, uploa, conjx, conjy, n, &alpha, x.data(), incx,
                                              y.data(), incy, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_syr2<T>( storage, uploa, conjx, conjy, n, alpha,
                           x.data(), incx, y.data(), incy, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, n, n, a.data(), a_ref.data(), lda, thresh );
}