#pragma once

#include "gemm.h"
#include "level3/ref_gemm.h"
#include "inc/check_error.h"
#include <stdexcept>
#include <algorithm>


template<typename T>
void test_gemm( char storage, char trnsa, char trnsb, gtint_t m, gtint_t n,
    gtint_t k, gtint_t lda_inc, gtint_t ldb_inc, gtint_t ldc_inc,
    T alpha, T beta, double thresh, char datatype ) {

    // Compute the leading dimensions of a, b, and c.
    gtint_t lda = testinghelpers::get_leading_dimension(storage, trnsa, m, k, lda_inc);
    gtint_t ldb = testinghelpers::get_leading_dimension(storage, trnsb, k, n, ldb_inc);
    gtint_t ldc = testinghelpers::get_leading_dimension(storage, 'n', m, n, ldc_inc);

    //----------------------------------------------------------
    //         Initialize matrics with random numbers
    //----------------------------------------------------------
    std::vector<T> a = testinghelpers::get_random_matrix<T>(-2, 8, storage, trnsa, m, k, lda, datatype);
    std::vector<T> b = testinghelpers::get_random_matrix<T>(-5, 2, storage, trnsb, k, n, ldb, datatype);
    std::vector<T> c = testinghelpers::get_random_matrix<T>(-3, 5, storage, 'n', m, n, ldc, datatype);

    // Create a copy of c so that we can check reference results.
    std::vector<T> c_ref(c);

    //----------------------------------------------------------
    //                  Call reference implementation.
    //----------------------------------------------------------
    testinghelpers::ref_gemm( storage, trnsa, trnsb, m, n, k, alpha,
               a.data(), lda, b.data(), ldb, beta, c_ref.data(), ldc );

    //----------------------------------------------------------
    //                  Call BLIS function
    //----------------------------------------------------------
    gemm<T>( storage, trnsa, trnsb, m, n, k, &alpha, a.data(), lda,
                                b.data(), ldb, &beta, c.data(), ldc );

    //----------------------------------------------------------
    //              check component-wise error.
    //----------------------------------------------------------
    computediff<T>( storage, m, n, c.data(), c_ref.data(), ldc, thresh );
}
