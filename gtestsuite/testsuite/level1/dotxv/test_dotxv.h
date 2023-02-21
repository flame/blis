#pragma once

#include "dotxv.h"
#include "level1/ref_dotxv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for dotxv operation.
 */

template<typename T>
static void test_dotxv( gtint_t n, char conjx, char conjy, T alpha,
  gtint_t incx, gtint_t incy, T beta, double thresh, char datatype )
{
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
    T rho_ref; 
    testinghelpers::initone(rho_ref);
    testinghelpers::ref_dotxv<T>(conjx, conjy, n, alpha, x.data(), incx, y.data(), incy, beta, &rho_ref);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    T rho; 
    testinghelpers::initone(rho);
    dotxv(conjx, conjy, n, &alpha, x.data(), incx, y.data(), incy, &beta, &rho);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( rho, rho_ref, thresh );
}