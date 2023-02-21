#pragma once

#include "amaxv.h"
#include "level1/ref_amaxv.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for amaxv operation.
 */

template<typename T>
void test_amaxv( gtint_t n, gtint_t incx, double thresh, char datatype ) {

    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x = testinghelpers::get_random_vector<T>(-10, 10, n, incx, datatype);

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    gtint_t idx_ref = testinghelpers::ref_amaxv<T>(n, x.data(), incx);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    gtint_t idx = amaxv(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    computediff<gtint_t>( idx, idx_ref );
}