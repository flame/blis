#pragma once

#include "her.h"
#include "level2/ref_her.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T, typename Tr>
void test_her( char storage, char uploa, char conjx, gtint_t n, Tr alpha,
               gtint_t incx, gtint_t lda_inc, double thresh, char datatype ) {

    // Compute the leading dimensions of a.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, 'n', n, n, lda_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random integer numbers.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 5, storage, 'n', n, n, lda, datatype);
    std::vector<T> x = testinghelpers::get_random_vector<T>(-3, 3, n, incx, datatype);

    mktrim<T>( storage, uploa, n, a.data(), lda );

    // Create a copy of c so that we can check reference results.
    std::vector<T> a_ref(a);
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    her<T,Tr>( storage, uploa, conjx, n, &alpha, x.data(), incx, a.data(), lda );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_her<T,Tr>( storage, uploa, conjx, n, alpha,
                                      x.data(), incx, a_ref.data(), lda );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, n, n, a.data(), a_ref.data(), lda, thresh );
}