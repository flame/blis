#pragma once

#include "symv.h"
#include "level2/ref_symv.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_symv( char storage, char uploa, char conja, char conjx, gtint_t n,
    T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
    double thresh, char datatype ) {

    // Compute the leading dimensions of a.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', n, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 5, storage, 'n', n, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(-3, 3, n, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(-2, 5, n, incy, datatype);

    mksymm<T>( storage, uploa, n, a.data(), lda );
    mktrim<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    symv<T>( storage, uploa, conja, conjx, n, &alpha, a.data(), lda,
                                  x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_symv<T>( storage, uploa, conja, conjx, n, &alpha,
                 a.data(), lda, x.data(), incx, &beta, y_ref.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy, thresh );
}