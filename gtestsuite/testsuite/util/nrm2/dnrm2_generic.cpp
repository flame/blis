#include <gtest/gtest.h>
#include "test_nrm2.h"

class dnrm2Test :
        public ::testing::TestWithParam<std::tuple<gtint_t, gtint_t, char>> {};

TEST_P( dnrm2Test, RandomData )
{
    using T = double;
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
    double thresh = std::sqrt(n)*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_nrm2<T>(n, incx, thresh, datatype);
}

// Prints the test case combination
class dnrm2TestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<gtint_t, gtint_t, char>> str) const {
        gtint_t n     = std::get<0>(str.param);
        gtint_t incx  = std::get<1>(str.param);
        char datatype = std::get<2>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "dnrm2_";
#elif TEST_CBLAS
        std::string str_name = "cblas_dnrm2";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_dnormfv";
#endif
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        str_name    = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        dnrm2Test,
        ::testing::Combine(
            ::testing::Range(gtint_t(10), gtint_t(101), 10),                 // m size of vector takes values from 10 to 100 with step size of 10.
            ::testing::Values(gtint_t(1), gtint_t(2)
#ifndef TEST_BLIS_TYPED
            ,gtint_t(-1), gtint_t(-2)
#endif
        ),                                                                   // stride size for x
            ::testing::Values('i')                                           // i : integer, f : float  datatype type tested
        ),
        ::dnrm2TestPrint()
    );
