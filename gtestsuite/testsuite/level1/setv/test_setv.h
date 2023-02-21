#pragma once

#include "setv.h"
#include "common/testing_helpers.h"
#include "inc/check_error.h"

/**
 * @brief Generic test body for setv operation.
 */

template<typename T>
void test_setv( char conjalpha, gtint_t n, T alpha, gtint_t incx ) {
    //----------------------------------------------------------
    //        Initialize vectors with random numbers.
    //----------------------------------------------------------
    std::vector<T> x( testinghelpers::buff_dim(n, incx), T{-1} );
    
    //----------------------------------------------------------
    //    Call reference implementation to get ref results.
    //----------------------------------------------------------
    T alpha_ref = alpha;
    if( testinghelpers::chkconj( conjalpha ) ) {
        alpha_ref = testinghelpers::conj<T>( alpha );
    }

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    setv( conjalpha, n, &alpha, x.data(), incx );

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    gtint_t i,idx;
    for( idx = 0 ; idx < n ; idx++ )
    {
        i = (incx > 0) ? (idx * incx) : ( - ( n - idx - 1 ) * incx );
        EXPECT_EQ(x[i], alpha_ref) << "blis_sol[" << i << "]="<< x[i] <<"   ref = "  << alpha_ref;
    }
}