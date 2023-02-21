#pragma once

#include "trsv.h"
#include "level2/ref_trsv.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_trsv( char storage, char uploa, char transa, char diaga, gtint_t n,
    T alpha, gtint_t lda_inc, gtint_t incx,  double thresh, char datatype ) {

    // Compute the leading dimensions for matrix size calculation.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, n, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(1, 5, storage, transa, n, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(1, 3, n, incx, datatype);

    mktrim<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> x_ref(x);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    trsv<T>( storage, uploa, transa, diaga, n, &alpha, a.data(), lda, x.data(), incx );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_trsv<T>( storage, uploa, transa, diaga, n, &alpha, a.data(), lda, x_ref.data(), incx );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, x.data(), x_ref.data(), incx, thresh );
}