#include <gtest/gtest.h>
#include "test_trmv.h"

class strmvTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   char,
                                                   char,
                                                   char,
                                                   gtint_t,
                                                   float,
                                                   gtint_t,
                                                   gtint_t,
                                                   char>> {};

TEST_P(strmvTest, RandomData) {
    using T = float;

    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // matrix storage format(row major, column major)
    char storage = std::get<0>(GetParam());
    // denotes whether matrix a is u,l
    char uploa = std::get<1>(GetParam());
    // denotes whether matrix a is n,c,t,h
    char transa = std::get<2>(GetParam());
    // denotes whether matrix diag is u,n
    char diaga = std::get<3>(GetParam());
    // matrix size n
    gtint_t n  = std::get<4>(GetParam());
    // specifies alpha value
    T alpha = std::get<5>(GetParam());
    // stride size for x:
    gtint_t incx = std::get<6>(GetParam());
    // lda increment.
    // If increment is zero, then the array size matches the matrix size.
    // If increment are nonnegative, the array size is bigger than the matrix size.
    gtint_t lda_inc = std::get<7>(GetParam());
    // specifies the datatype for randomgenerators
    char datatype   = std::get<8>(GetParam());

    // Set the threshold for the errors:
    double thresh = 10*n*testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call test body using these parameters
    //----------------------------------------------------------
    test_trmv<T>(storage, uploa, transa, diaga, n, alpha, lda_inc, incx, thresh, datatype);
}

class strmvTestPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,char,char,gtint_t,float,gtint_t,gtint_t,char>> str) const {
        char sfm       = std::get<0>(str.param);
        char uploa     = std::get<1>(str.param);
        char transa    = std::get<2>(str.param);
        char diaga     = std::get<3>(str.param);
        gtint_t n      = std::get<4>(str.param);
        float alpha    = std::get<5>(str.param);
        gtint_t incx   = std::get<6>(str.param);
        gtint_t ld_inc = std::get<7>(str.param);
        char datatype  = std::get<8>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "strmv_";
#elif TEST_CBLAS
        std::string str_name = "cblas_strmv";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "blis_strmv";
#endif
        str_name    = str_name + "_" + sfm;
        str_name    = str_name + "_" + uploa+transa;
        str_name    = str_name + "_d" + diaga;
        str_name    = str_name + "_" + std::to_string(n);
        std::string alpha_str = ( alpha > 0) ? std::to_string(int(alpha)) : ("m" + std::to_string(int(std::abs(alpha))));
        str_name    = str_name + "_a" + alpha_str;
        std::string incx_str = ( incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name    = str_name + "_" + incx_str;
        str_name    = str_name + "_" + std::to_string(ld_inc);
        str_name    = str_name + "_" + datatype;
        return str_name;
    }
};

// Black box testing.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        strmvTest,
        ::testing::Combine(
            ::testing::Values('c'
#ifndef TEST_BLAS
            ,'r'
#endif
            ),                                                               // storage format
            ::testing::Values('u','l'),                                      // uploa
            ::testing::Values('n','t'),                                      // transa
            ::testing::Values('n','u'),                                      // diaga , n=NONUNIT_DIAG u=UNIT_DIAG
            ::testing::Range(gtint_t(10), gtint_t(31), 10),                  // n
            ::testing::Values( 1.0
#ifdef TEST_BLIS_TYPED
            , -2.0
#endif
            ),                                                               // alpha
            ::testing::Values(gtint_t(1)),                                   // stride size for x
            ::testing::Values(gtint_t(0), gtint_t(1)),                       // increment to the leading dim of a
            ::testing::Values(ELEMENT_TYPE)                                  // i : integer, f : float  datatype type tested
        ),
        ::strmvTestPrint()
    );
