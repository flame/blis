#pragma once

#include "trmm3.h"
#include "level3/ref_trmm3.h"
#include "inc/check_error.h"
#include "inc/utils.h"
#include <stdexcept>
#include <algorithm>

template<typename T>
void test_trmm3( char storage, char side, char uploa, char transa, char diaga,
  char transb, gtint_t m, gtint_t n, T alpha, gtint_t lda_inc, gtint_t ldb_inc,
  T beta, gtint_t ldc_inc, double thresh, char datatype ) {

    gtint_t mn;
    testinghelpers::set_dim_with_side( side, m, n, &mn );
    gtint_t lda = testinghelpers::get_leading_dimension(storage, transa, mn, mn, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, transb, m, n, ldb_inc);
    gtint_t ldc = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldc_inc);

    //----------------------------------------------------------
    //        Initialize matrics with random values.
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 8, storage, transa, mn, mn, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-5, 2, storage, transb, m, n, ldb, datatype);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-3, 5, storage, 'n', m, n, ldc, datatype);

    // Create a copy of v so that we can check reference results.
    std::vector<T> c_ref(c);

    mktrim<T>( storage, uploa, mn, a.data(), lda );
    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    trmm3<T>( storage, side, uploa, transa, diaga, transb, m, n, &alpha,
                    a.data(), lda, b.data(), ldb, &beta, c.data(), ldc );
    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_trmm3( storage, side, uploa, transa, diaga, transb,
          m, n, alpha, a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldb, thresh );
}