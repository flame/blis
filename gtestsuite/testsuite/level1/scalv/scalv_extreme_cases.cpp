#include <gtest/gtest.h>
#include "test_scalv.h"

template <typename T>
class xscalv : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam;
TYPED_TEST_SUITE(xscalv, TypeParam);

TYPED_TEST(xscalv, zero_alpha_x_fp)
{
    using T = TypeParam;
    gtint_t n = 10, incx = 1;
    std::vector<T> x(n);
    // Initialize x with random numbers.
    testinghelpers::datagenerators::randomgenerators(n, incx, x.data(), 'f');
    std::vector<T> x_ref(x);
    T alpha = T{0};

    testinghelpers::ref_scalv<T>('n', n, alpha, x_ref.data(), incx);
    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    scalv<T>('n', n, alpha, x.data(), incx);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();
    computediff<T>( n, x.data(), x_ref.data(), incx, thresh );
}

TYPED_TEST(xscalv, zero_alpha_x_inf)
{
    using T = TypeParam;
    gtint_t n = 10, incx = 1;
    std::vector<T> x(n);
    // Initialize x with random numbers.
    testinghelpers::datagenerators::randomgenerators(n, incx, x.data(), 'f');
    x[3] = 1.0/0.0;
    std::vector<T> x_ref(x);
    T alpha = T{0};
    testinghelpers::ref_scalv<T>('n', n, alpha, x_ref.data(), incx);

    //----------------------------------------------------------
    //                  Call BLIS function.
    //----------------------------------------------------------
    scalv<T>('n', n, alpha, x.data(), incx);

    //----------------------------------------------------------
    //              Compute component-wise error.
    //----------------------------------------------------------
    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();
    computediff<T>( n, x.data(), x_ref.data(), incx, thresh );
}