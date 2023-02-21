#include <gtest/gtest.h>
#include "test_addv.h"

class ZAddvGenericTest :
        public ::testing::TestWithParam<std::tuple<char, gtint_t, gtint_t, gtint_t, char>> {};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ZAddvGenericTest);

TEST_P( ZAddvGenericTest, RandomData )
{
    using T = dcomplex;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes whether x or conj(x) will be added to y:
    char conj_x = std::get<0>(GetParam());
    // vector length:
    gtint_t n = std::get<1>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<2>(GetParam());
    // stride size for y:
    gtint_t incy = std::get<3>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype = std::get<4>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_addv<T>(conj_x, n, incx, incy, thresh, datatype);
}

// Prints the test case combination
class ZAddvGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,gtint_t,gtint_t,gtint_t,char>> str) const {
        char conj      = std::get<0>(str.param);
        gtint_t n      = std::get<1>(str.param);
        gtint_t incx   = std::get<2>(str.param);
        gtint_t incy   = std::get<3>(str.param);
        char datatype  = std::get<4>(str.param);
        std::string str_name = "bli_zaddv";
        str_name += "_" + std::to_string(n);
        str_name += "_" + std::string(&conj, 1);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        std::string incy_str = ( incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_" + incy_str;
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};

#ifdef TEST_BLIS_TYPED
// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        ZAddvGenericTest,
        ::testing::Combine(
            ::testing::Values('n','c'),                                      // n: not transpose for x, c: conjugate for x
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(1)),                                   // stride size for y
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::ZAddvGenericTestPrint()
    );
#endif