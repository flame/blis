#pragma once

#include "dotv.h"
#include "level1/ref_dotv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for dotv operation.
 */

template<typename T>
static void test_dotv( char conjx, char conjy, gtint_t n, gtint_t incx,
    gtint_t incy, double thresh, char datatype )
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
    if constexpr (testinghelpers::type_info<T>::is_real)
        testinghelpers::ref_dotv<T>( n, x.data(), incx, y_ref.data(), incy, &rho_ref );
    else
        testinghelpers::ref_dotv<T>(conjx, conjy, n, x.data(), incx, y_ref.data(), incy, &rho_ref);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    T rho;
    dotv<T>(conjx, conjy, n, x.data(), incx, y.data(), incy, &rho);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<T>( rho, rho_ref, thresh );
}