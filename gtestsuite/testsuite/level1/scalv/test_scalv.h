#pragma once

#include "scalv.h"
#include "level1/ref_scalv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for axpby operation.
 */

template<typename T>
static void test_scalv(char conja_alpha, gtint_t n, gtint_t incx, T alpha, double thresh, char datatype)
{
    //----------------------------------------------------------
    //        Initialize vector with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>(-10, 10, n, incx, datatype);

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    std::vector<T> x_ref(x);
    testinghelpers::ref_scalv<T>(conja_alpha, n, alpha, x_ref.data(), incx);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    scalv<T>(conja_alpha, n, alpha, x.data(), incx);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<T>( n, x.data(), x_ref.data(), incx, thresh );
}