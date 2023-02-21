#pragma once

#include "copyv.h"
#include "level1/ref_copyv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for copyv operation.
 */

template<typename T>
static void test_copyv( char conjx, gtint_t n, gtint_t incx, gtint_t incy,
                                             double thresh, char datatype ) {

    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>(-10, 10, n, incx, datatype);
    std::vector<T> y( testinghelpers::buff_dim(n, incy), T{-1} );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> y_ref(y);

    testinghelpers::ref_copyv<T>(conjx, n, x.data(), incx, y_ref.data(), incy);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    copyv<T>(conjx, n, x.data(), incx, y.data(), incy);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy );
}