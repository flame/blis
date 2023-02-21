#include <gtest/gtest.h>
#include "test_amaxv.h"

class zamaxvGenericTest :
        public ::testing::TestWithParam<std::tuple<gtint_t,
                                                   gtint_t,
                                                   char>> {};

// Tests using random integers as vector elements.
TEST_P( zamaxvGenericTest, RandomData )
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
    // specifies the datatype for randomgenerators
    char datatype = std::get<2>(GetParam());

    // Set the threshold for the errors:
    double thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_amaxv<T>(n, incx, thresh, datatype);
}

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class zamaxvGenericTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t,gtint_t,char>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
        char datatype = std::get<2>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "izamax_";
#elif TEST_CBLAS
        std::string str_name = "cblas_izamax";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_zamaxv";
#endif
        str_name += "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        str_name = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing for generic and main use of zamaxv.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        zamaxvGenericTest,
        ::testing::Combine(
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::zamaxvGenericTestPrint()
    );

// Test for non-unit increments.
// Only test very few cases as sanity check.
// We can modify the values using implementantion details.
INSTANTIATE_TEST_SUITE_P(
        NonUnitIncrements,
        zamaxvGenericTest,
        ::testing::Combine(
            ::testing::Values(gtint_t(3), gtint_t(30), gtint_t(112)),        // m size of vector
            ::testing::Values(gtint_t(2), gtint_t(11)),                      // stride size for x
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::zamaxvGenericTestPrint()
    );
