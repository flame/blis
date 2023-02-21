#pragma once

#include "nrm2.h"
#include "util/ref_nrm2.h"
#include "inc/check_error.h"

template<typename T>
void test_nrm2( gtint_t n, gtint_t incx, double thresh, char datatype )
{
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x( testinghelpers::buff_dim(n, incx) );
    testinghelpers::datagenerators::randomgenerators( -10, 10, n, incx, x.data(), datatype );

    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    // Create a copy of y so that we can check reference results.
    using real = typename testinghelpers::type_info<T>::real_type;
    real norm_ref = testinghelpers::ref_nrm2<T, real>( n, x.data(), incx );

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    real norm = nrm2<T, real>(n, x.data(), incx);

    //----------------------------------------------------------
    //              Compute error.
    //----------------------------------------------------------
    computediff<real>( norm, norm_ref, thresh );
}

