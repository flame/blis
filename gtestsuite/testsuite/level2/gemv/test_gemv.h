#pragma once

#include "gemv.h"
#include "level2/ref_gemv.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>

template<typename T>

void test_gemv( char storage, char trnsa, char conjx, gtint_t m, gtint_t n,
    T alpha, gtint_t lda_inc, gtint_t incx, T beta, gtint_t incy,
    double thresh, char datatype ) {

    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', m, n, lda_inc);

    // Get correct vector lengths.
    gtint_t lenx = ( testinghelpers::chknotrans( trnsa ) ) ? n : m ;
    gtint_t leny = ( testinghelpers::chknotrans( trnsa ) ) ? m : n ;

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(1, 5, storage, 'n', m, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(1, 3, lenx, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(1, 3, leny, incy, datatype);

    // Create a copy of c so that we can check reference results.
    std::vector<T> y_ref(y);

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemv( storage, trnsa, conjx, m, n, alpha, a.data(),
                         lda, x.data(), incx, beta, y_ref.data(), incy );

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemv( storage, trnsa, conjx, m, n, &alpha, a.data(), lda,
                         x.data(), incx, &beta, y.data(), incy );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( leny, y.data(), y_ref.data(), incy, thresh );
}