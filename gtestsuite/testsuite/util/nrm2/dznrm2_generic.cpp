#include <gtest/gtest.h>
#include "test_nrm2.h"

class dznrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t>> {};

TEST_P( dznrm2Test, RandomData )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // vector length:
    gtint_t n = std::get<0>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<1>(GetParam());

    // Set the threshold for the errors:
    double thresh = 3*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, thresh);
}

// Prints the test case combination
class dznrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dznrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dznrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_znormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        return str_name;
    }
};

/**
 * dznrm2 implementation is composed by two parts:
 * - vectorized path for n>2
 *      - for-loop for multiples of 4 (F4)
 *      - for-loop for multiples of 2 (F2)
 * - scalar path for n<=2 (S)
*/
INSTANTIATE_TEST_SUITE_P(
        AT,
        dznrm2Test,
        ::testing::Combine(
            // m size of vector
            ::testing::Values(gtint_t(1),  // trivial case n=1
                              gtint_t(2),  // will only go through S
                              gtint_t(4),  // 1*4 - will only go through F4
                              gtint_t(12), // 3*4 - will go through F4
                              gtint_t(17), // 4*4 + 1 - will go through F4 & S
                              gtint_t(22), // 5*4 + 2 - will go through F4 & F2
                              gtint_t(35), // 8*4 + 2 + 1 - will go through F4 & F2 & S
                              gtint_t(78), // a few bigger numbers
                              gtint_t(112),
                              gtint_t(187),
                              gtint_t(213)
            ),
            // stride size for x
            ::testing::Values(gtint_t(1), gtint_t(3)
#ifndef TEST_BLIS_TYPED
            , gtint_t(-1), gtint_t(-7)
#endif
        )
        ),
        ::dznrm2TestPrint()
    );