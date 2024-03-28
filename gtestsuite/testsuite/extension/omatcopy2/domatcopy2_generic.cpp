/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <gtest/gtest.h>
#include "test_omatcopy2.h"

class domatcopy2API :
        public ::testing::TestWithParam<std::tuple<char,        // storage
                                                   char,        // trans
                                                   gtint_t,     // m
                                                   gtint_t,     // n
                                                   double,      // alpha
                                                   gtint_t,     // lda_inc
                                                   gtint_t,     // stridea
                                                   gtint_t,     // ldb_inc
                                                   gtint_t,     // strideb
                                                   bool>> {};   // is_memory_test

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(domatcopy2API);

// Tests using random numbers as vector elements.
TEST_P( domatcopy2API, FunctionalTest )
{
    using T = double;
    //----------------------------------------------------------
    // Initialize values from the parameters passed through
    // test suite instantiation (INSTANTIATE_TEST_SUITE_P).
    //----------------------------------------------------------
    // denotes the storage format of the input matrices
    char storage = std::get<0>(GetParam());
    // denotes the trans value for the operation
    char trans = std::get<1>(GetParam());
    // m dimension
    gtint_t m = std::get<2>(GetParam());
    // n dimension
    gtint_t n = std::get<3>(GetParam());
    // alpha
    T alpha = std::get<4>(GetParam());
    // lda_inc for A
    gtint_t lda_inc = std::get<5>(GetParam());
    // stridea
    gtint_t stridea = std::get<6>(GetParam());
    // ldb_inc for B
    gtint_t ldb_inc = std::get<7>(GetParam());
    // strideb
    gtint_t strideb = std::get<8>(GetParam());
    // is_memory_test
    bool is_memory_test = std::get<9>(GetParam());

    double thresh = 0.0;
    // Set the threshold for the errors
    if( ( alpha != testinghelpers::ZERO<T>() || alpha != testinghelpers::ONE<T>() ) )
      thresh = testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_omatcopy2<T>( storage, trans, m, n, alpha, lda_inc, stridea, ldb_inc, strideb, thresh, is_memory_test );
}

// Test-case logger : Used to print the test-case details based on parameters
// The string format is as follows :
// {blas_/cblas_/bli_}_storage_trans_m_n_alpha_lda_ldb_{mem_test_enabled/mem_test_disabled}
class domatcopy2APIPrint {
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char,char,gtint_t,gtint_t,double,gtint_t,gtint_t,gtint_t,gtint_t,bool>> str) const {
        char storage   = std::get<0>(str.param);
        char trans     = std::get<1>(str.param);
        gtint_t m      = std::get<2>(str.param);
        gtint_t n      = std::get<3>(str.param);
        double alpha    = std::get<4>(str.param);
        gtint_t lda_inc = std::get<5>(str.param);
        gtint_t stridea = std::get<6>(str.param);
        gtint_t ldb_inc = std::get<7>(str.param);
        gtint_t strideb = std::get<8>(str.param);
        bool is_memory_test = std::get<9>(str.param);
// Currently, BLIS only has the BLAS standard wrapper for this API.
// The CBLAS and BLIS strings are also added here(with macro guards),
// in case we add the CBLAS and BLIS wrappers to the library in future.
#ifdef TEST_BLAS
        std::string str_name = "blas_";
#elif TEST_CBLAS
        std::string str_name = "cblas_";
#else  //#elif TEST_BLIS_TYPED
        std::string str_name = "bli_";
#endif
        str_name += std::string(&storage, 1);
        str_name += "_" + std::string(&trans, 1);
        str_name += "_" + std::to_string(m);
        str_name += "_" + std::to_string(n);
        std::string alpha_str = ( alpha >= 0) ? std::to_string(int(alpha)) : ("m" + std::to_string(int(std::abs(alpha))));
        str_name = str_name + "_a" + alpha_str;
        gtint_t lda = testinghelpers::get_leading_dimension( storage, 'n', m, n, lda_inc );
        gtint_t ldb = testinghelpers::get_leading_dimension( storage, trans, m, n, ldb_inc );
        str_name += "_lda" + std::to_string(lda);
        str_name += "_stridea" + std::to_string(stridea);
        str_name += "_ldb" + std::to_string(ldb);
        str_name += "_strideb" + std::to_string(strideb);
        str_name += ( is_memory_test )? "_mem_test_enabled" : "_mem_test_disabled";

        return str_name;
    }
};

#if defined(TEST_BLAS) && defined(REF_IS_MKL)
// Black box testing for generic and main use of domatcopy2.
INSTANTIATE_TEST_SUITE_P(
        Blackbox,
        domatcopy2API,
        ::testing::Combine(
            ::testing::Values('c'),                                          // storage format(currently only for BLAS testing)
            ::testing::Values('n', 't', 'r', 'c'),                           // trans(and/or conj) value
                                                                             // 'n' - no-transpose, 't' - transpose
                                                                             // 'r' - conjugate,    'c' - conjugate-transpose
            ::testing::Values(gtint_t(10), gtint_t(55), gtint_t(243)),       // m
            ::testing::Values(gtint_t(10), gtint_t(55), gtint_t(243)),       // n
            ::testing::Values(2.0, -3.0, 1.0, 0.0),                          // alpha
            ::testing::Values(gtint_t(0), gtint_t(25)),                      // increment of lda
            ::testing::Values(gtint_t(1), gtint_t(3)),                       // stridea
            ::testing::Values(gtint_t(0), gtint_t(25)),                      // increment of ldb
            ::testing::Values(gtint_t(1), gtint_t(3)),                       // strideb
            ::testing::Values(false, true)                                   // is_memory_test
        ),
        ::domatcopy2APIPrint()
    );
#endif
