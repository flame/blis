#include <gtest/gtest.h>
#include "test_nrm2.h"

/**
 * Testing edge input parameters.
 * 
 * zero n should return 0.
 * zero incx should return sqrt(n*abs(x[0])**2).
*/

// Early return.
template <typename T>
class nrm2_ERS : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(nrm2_ERS, TypeParam);

TYPED_TEST(nrm2_ERS, zero_n) {
    using T = TypeParam;    
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 0;
    gtint_t incx = 1;
    // initialize norm to ensure that it is set to zero from nrm2 and it does not simply return.
    RT blis_norm = 19.0; 
    // using nullptr since x should not be accessed anyway.
    // If "x" is accessed before return then nrm2 would segfault.
    blis_norm = nrm2<T>(n, nullptr, incx);
    RT ref_norm = testinghelpers::ref_nrm2<T>(n, nullptr, incx);
    computediff<RT>(blis_norm, ref_norm);
}

// Edge case where it actually does not return early.
// Since there are 2 different paths, vectorized and scalar,
// we break this into 2 tests, once for each case.
template <typename T>
class nrm2_EIC : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(nrm2_EIC, TypeParam);

TYPED_TEST(nrm2_EIC, zero_incx_scalar) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;    
    gtint_t n = 2;    
    gtint_t incx = 0;
    std::vector<T> x(n);
    for (auto &xi : x)
        testinghelpers::initone(xi);
    // For incx=0, nrm2 iterates through the first element n-times.
    // So, we initialize x[0] with a different value than the rest
    // of the elements.
    x[0] = T{2.0}*x[0];
    RT blis_norm = 19.0;
    blis_norm = nrm2<T>(n, x.data(), incx);
    RT ref_norm = testinghelpers::ref_nrm2<T>(n, x.data(), incx);
    computediff<RT>(blis_norm, ref_norm);
}

TYPED_TEST(nrm2_EIC, zero_incx_vectorized) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;    
    gtint_t n = 64;    
    gtint_t incx = 0;
    std::vector<T> x(n);
    for (auto &xi : x)
        testinghelpers::initone(xi);
    // For incx=0, nrm2 iterates through the first element n-times.
    // So, we initialize x[0] with a different value than the rest
    // of the elements.
    x[0] = T{2.0}*x[0];
    RT blis_norm = 19.0;
    blis_norm = nrm2<T>(n, x.data(), incx);
    RT ref_norm = testinghelpers::ref_nrm2<T>(n, x.data(), incx);
    computediff<RT>(blis_norm, ref_norm);
}

/*
    The following test is specific to dnrm2 and dznrm2 apis.
    In case of multithreading, each thread will calculate its
    norm based on the data it operates on. All these norms will
    be reduced post the parallel region.
*/
TYPED_TEST( nrm2_EIC, zero_incx_MT ) {
    using T = TypeParam;
    using RT = typename testinghelpers::type_info<T>::real_type;
    gtint_t n = 2950000;
    gtint_t incx = 0;
    std::vector<T> x(n);
    for (auto &xi : x)
        testinghelpers::initone(xi);
    // For incx=0, nrm2 iterates through the first element n-times.
    // So, we initialize x[0] with a different value than the rest
    // of the elements.
    x[0] = T{2.0}*x[0];
    RT blis_norm = nrm2<T>(n, x.data(), incx);
    RT ref_norm = testinghelpers::ref_nrm2<T>(n, x.data(), incx);
    computediff<RT>(blis_norm, ref_norm);
}
