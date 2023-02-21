#include <gtest/gtest.h>
#include "test_syr.h"

class ssyrTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   float,
                                                   gtint_t,
                                                   gtint_t,
                                                   char>> {};

TEST_P(ssyrTest, RandomData) {
    using T = float;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is u,l
    char uploa = std::get<1>(GetParam());
    // denotes whether vector x is n,c
    char conjx = std::get<2>(GetParam());
    // matrix size n
    gtint_t n  = std::get<3>(GetParam());
    // specifies alpha value
    float alpha = std::get<4>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<5>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<6>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype   = std::get<7>(GetParam());

    // Set the threshold for the errors:
    double thresh = 2*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_syr<T>(storage, uploa, conjx, n, alpha, incx, lda_inc, thresh, datatype);
}

class ssyrTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,gtint_t,float,gtint_t,gtint_t,char>> str) const {
        char sfm       = std::get<0>(str.param);
        char uploa     = std::get<1>(str.param);
        char conjx     = std::get<2>(str.param);
        gtint_t n      = std::get<3>(str.param);
        float alpha    = std::get<4>(str.param);
        gtint_t incx   = std::get<5>(str.param);
        gtint_t ld_inc = std::get<6>(str.param);
        char datatype  = std::get<7>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "ssyr_";
#elif TEST_CBLAS
        std::string str_name = "cblas_ssyr";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_ssyr";
#endif
        str_name    = str_name + "_" + sfm;
        str_name    = str_name + "_" + uploa+conjx;
        str_name    = str_name + "_" + std::to_string(n);
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        std::string alpha_str = ( alpha > 0) ? std::to_string(int(alpha)) : ("m" + std::to_string(int(std::abs(alpha))));
        str_name    = str_name + "_a" + alpha_str;
        str_name    = str_name + "_" + std::to_string(ld_inc);
        str_name    = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        ssyrTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n'),                                          // conjx
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // n
            ::testing::Values(1.0),                                          // alpha
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(0), gtint_t(3)),                       // increment to the leading dim of a
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::ssyrTestPrint()
    );
