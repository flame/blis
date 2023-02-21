#include <gtest/gtest.h>
#include "test_nrm2.h"

template <typename T>
class xnrm2 : public ::testing::Test {};
typedef ::testing::Types<float, double> TypeParam;
TYPED_TEST_SUITE(xnrm2, TypeParam);

TYPED_TEST(xnrm2, zeroFP) {
    using T = TypeParam;
    T x = T(0);

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(0, norm);
}

TYPED_TEST(xnrm2, minFP) {
    using T = TypeParam;
    T x = std::numeric_limits<T>::min();

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(x, norm);
}

TYPED_TEST(xnrm2, maxFP) {
    using T = TypeParam;
    T x = std::numeric_limits<T>::max();

    T norm = nrm2<T,T>(1, &x, 1);
    EXPECT_EQ(x, norm);
}

TEST(dnrm2, largeDouble) {
    using T = double;
    gtint_t n = 2;
    std::vector<T> x{3e300, 4e300}, y{-4e300, -3e300};

    T norm = nrm2<T,T>(n, x.data(), 1);
    EXPECT_EQ(5e300, norm);

    norm = nrm2<T,T>(n, y.data(), 1);
    EXPECT_EQ(5e300, norm);
}
