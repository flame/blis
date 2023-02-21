#pragma once

#include "axpbyv.h"
#include "level1/ref_axpbyv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for axpby operation.
 */

template<typename T>
static void test_axpbyv( char conjx, gtint_t n, gtint_t incx, gtint_t incy,
    T alpha, T beta, double thresh, char datatype ) {

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
    testinghelpers::ref_axpbyv<T>(conjx, n, alpha, x.data(), incx, beta, y_ref.data(), incy);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    axpbyv<T>(conjx, n, alpha, x.data(), incx, beta, y.data(), incy);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy, thresh );
}