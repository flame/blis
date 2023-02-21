#pragma once

#include "axpyv.h"
#include "level1/ref_axpyv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for axpyv operation.
 */

template<typename T>
static void test_axpyv( char conjx, gtint_t n, gtint_t incx, gtint_t incy,
    T alpha, double thresh, char datatype ) {
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>(-10, 10, n, incx, datatype);
    std::vector<T> y = testinghelpers::get_random_vector<T>(-10, 10, n, incy, datatype);

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);
    testinghelpers::ref_axpyv<T>(conjx, n, alpha, x.data(), incx, y_ref.data(), incy);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    axpyv<T>(conjx, n, alpha, x.data(), incx, y.data(), incy);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy, thresh );
}