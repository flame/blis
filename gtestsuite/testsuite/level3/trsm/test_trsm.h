#pragma once

#include "trsm.h"
#include "level3/ref_trsm.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_trsm( char storage, char side, char uploa, char transa,
    char diaga, gtint_t m, gtint_t n, T alpha, gtint_t lda_inc,
    gtint_t ldb_inc, double thresh, char datatype ) {

    gtint_t mn;
    testinghelpers::set_dim_with_side( side, m, n, &mn );
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, mn, mn, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldb_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random values.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(2, 18, storage, transa, mn, mn, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(5, 12, storage, 'n', m, n, ldb, datatype);

    // Create a copy of v so that we can check reference results.
    std::vector<T> b_ref(b);

    mktrim<T>( storage, uploa, mn, a.data(), lda );
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    trsm<T>( storage, side, uploa, transa, diaga, m, n, &alpha, a.data(), lda, b.data(), ldb );

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_trsm( storage, side, uploa, transa, diaga, m, n, alpha, a.data(), lda, b_ref.data(), ldb );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, b.data(), b_ref.data(), ldb, thresh );
}