#pragma once

#include "subv.h"
#include "level1/ref_subv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for subv operation.
 */

template<typename T>
void test_subv( char conjx, gtint_t n, gtint_t incx, gtint_t incy,
               double thresh, char datatype ) {
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
    testinghelpers::ref_subv<T>(conjx, n, x.data(), incx, y_ref.data(), incy);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    subv(conjx, n, x.data(), incx, y.data(), incy);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, y.data(), y_ref.data(), incy, thresh );

}