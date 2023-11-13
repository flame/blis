/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "test_axpbyv.h"

class zaxpbyvEVTTest :
        public ::testing::TestWithParam<std::tuple<char,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   gtint_t,
                                                   dcomplex,
                                                   gtint_t,
                                                   dcomplex,
                                                   dcomplex,
                                                   dcomplex>> {};
// Tests using random integers as vector elements.
TEST_P(zaxpbyvEVTTest, RandomData)
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
    // index for exval in x
    gtint_t xi = std::get<4>(GetParam());
    // exval for x
    T xexval = std::get<5>(GetParam());
    // index for exval in y
    gtint_t yj = std::get<6>(GetParam());
    // exval for x
    T yexval = std::get<7>(GetParam());
    // alpha
    T alpha = std::get<8>(GetParam());
    // beta
    T beta = std::get<9>(GetParam());

    // Set the threshold for the errors:
    double thresh = 20 * testinghelpers::getEpsilon<T>();

    //----------------------------------------------------------
    //     Call generic test body using those parameters
    //----------------------------------------------------------
    test_axpbyv<T>(conj_x, n, incx, incy, alpha, beta, xi, xexval,
                   yj, yexval, thresh);
}

// Used to generate a test case with a sensible name.
// Beware that we cannot use fp numbers (e.g., 2.3) in the names,
// so we are only printing int(2.3). This should be enough for debugging purposes.
// If this poses an issue, please reach out.
class zaxpbyvEVTVecPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, gtint_t, gtint_t, gtint_t, gtint_t, dcomplex, gtint_t, dcomplex, dcomplex, dcomplex>> str) const
    {
        char conj = std::get<0>(str.param);
        gtint_t n = std::get<1>(str.param);
        gtint_t incx = std::get<2>(str.param);
        gtint_t incy = std::get<3>(str.param);
        gtint_t xi = std::get<4>(str.param);
        dcomplex xexval = std::get<5>(str.param);
        gtint_t yj = std::get<6>(str.param);
        dcomplex yexval = std::get<7>(str.param);
        dcomplex alpha = std::get<8>(str.param);
        dcomplex beta = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "zaxpby_";
#elif TEST_CBLAS
        std::string str_name = "cblas_zaxpby";
#else // #elif TEST_BLIS_TYPED
        std::string str_name = "bli_zaxpbyv";
#endif
        str_name += "_" + std::to_string(n);
        str_name += "_" + std::string(&conj, 1);
        std::string incx_str = (incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        std::string incy_str = (incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_" + incy_str;
        std::string xexval_str = testinghelpers::get_value_string(xexval);
        std::string yexval_str = testinghelpers::get_value_string(yexval);
        str_name = str_name + "_X_" + std::to_string(xi);
        str_name = str_name + "_" + xexval_str;
        str_name = str_name + "_Y_" + std::to_string(yj);
        str_name = str_name + "_" + yexval_str;
        std::string alpha_str = testinghelpers::get_value_string(alpha);
        std::string beta_str = testinghelpers::get_value_string(beta);
        str_name = str_name + "_a" + alpha_str;
        str_name = str_name + "_b" + beta_str;
        return str_name;
    }
};

class zaxpbyvAlphaBetaPrint
{
public:
    std::string operator()(
        testing::TestParamInfo<std::tuple<char, gtint_t, gtint_t, gtint_t, gtint_t, dcomplex, gtint_t, dcomplex, dcomplex, dcomplex>> str) const
    {
        char conj = std::get<0>(str.param);
        gtint_t n = std::get<1>(str.param);
        gtint_t incx = std::get<2>(str.param);
        gtint_t incy = std::get<3>(str.param);
        dcomplex alpha = std::get<8>(str.param);
        dcomplex beta = std::get<9>(str.param);
#ifdef TEST_BLAS
        std::string str_name = "zaxpby_";
#elif TEST_CBLAS
        std::string str_name = "cblas_zaxpby";
#else // #elif TEST_BLIS_TYPED
        std::string str_name = "bli_zaxpbyv";
#endif
        str_name += "_" + std::to_string(n);
        str_name += "_" + std::string(&conj, 1);
        std::string incx_str = (incx > 0) ? std::to_string(incx) : "m" + std::to_string(std::abs(incx));
        str_name += "_" + incx_str;
        std::string incy_str = (incy > 0) ? std::to_string(incy) : "m" + std::to_string(std::abs(incy));
        str_name += "_" + incy_str;
        std::string alpha_str = testinghelpers::get_value_string(alpha);
        std::string beta_str = testinghelpers::get_value_string(beta);
        str_name = str_name + "_a" + alpha_str;
        str_name = str_name + "_b" + beta_str;
        return str_name;
    }
};

static double NaN = std::numeric_limits<double>::quiet_NaN();
static double Inf = std::numeric_limits<double>::infinity();

/*
    The code structure for bli_zaxpbyv_zen_int( ... ) is as follows :
    For unit strides :
        Main loop    :  In blocks of 8 --> L8
        Fringe loops :  In blocks of 6 --> L6
                        In blocks of 4 --> L4
                        In blocks of 2 --> L2

    For non-unit strides : A single loop, to process element wise.
    NOTE : Any size, requiring the fringe case of 1 with unit stride falls to
           the non-unit stride loop and executes it once for just the last element.

    With regards to exception value testing, every loop is tested separately.
    The indices for setting exception values on the vectors are such that
    every load associated with the loop has an exception value in it. Thus,
    every arithmetic instruction associated with each load will be tested
    for exception value handling.
*/

// Exception value testing(on vectors) for L8
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_vec_L8,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(8)), // m, size of vector to enter L8 directly.
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(4), gtint_t(7)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(5), gtint_t(6)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{0.9, 4.5}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on vectors) for L6
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_vec_L6,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(6)), // m, size of vector to enter L8 directly.
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(3), gtint_t(4)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(2), gtint_t(5)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{0.9, 4.5}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on vectors) for L4
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_vec_L4,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(4)), // m, size of vector to enter L8 directly.
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(gtint_t(1), gtint_t(3)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on x
        ::testing::Values(gtint_t(0), gtint_t(2)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{0.9, 4.5}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on vectors) for L2
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_vec_L2,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(2)), // m, size of vector to enter L8 directly.
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(gtint_t(1)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on x
        ::testing::Values(gtint_t(0)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN},
                          dcomplex{NaN, -Inf}), // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{0.9, 4.5}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on vectors) with non unit strides
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_vec_NUS,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(1), gtint_t(5)), // m, size of vector to enter NUS loop directly.
        ::testing::Values(gtint_t(3)), // stride size for x
        ::testing::Values(gtint_t(-4)), // stride size for y
        ::testing::Values(gtint_t(0)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN}), // exception values to set on x
        ::testing::Values(gtint_t(0)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{NaN, 2.3},
                          dcomplex{-Inf, 0.0}, dcomplex{Inf, NaN}), // exception values to set on y
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{2.2, -3.3}), // alpha
        ::testing::Values(dcomplex{0.0, 0.0}, dcomplex{1.0, 0.0}, dcomplex{0.9, 4.5}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on alpha/beta) with unit stride
/*
    NOTE : Here, every loop is tested for, with alpha and beta having exception values
           Furthermore, the first element of x and second element of y are set to 0, which
           includes testing that cover cases where NaN might be induced due to 0 * (Inf or -Inf).
*/
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_alphabeta_US,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(8), gtint_t(6), gtint_t(4), gtint_t(2)), // m size of vector to enter L8, L6, L4 and L2 respectively.
        ::testing::Values(gtint_t(1)), // stride size for x
        ::testing::Values(gtint_t(1)), // stride size for y
        ::testing::Values(gtint_t(0)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}), // exception values to set on x
        ::testing::Values(gtint_t(1)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}), // exception values to set on y
        ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0}, dcomplex{-Inf, NaN}), // alpha
        ::testing::Values(dcomplex{-0.9, NaN}, dcomplex{0.0, -Inf}, dcomplex{NaN, Inf}) // beta
        ),
    ::zaxpbyvEVTVecPrint());

// Exception value testing(on alpha/beta) with non-unit stride
INSTANTIATE_TEST_SUITE_P(
    bli_zaxpbyv_zen_int_evt_alphabeta_NUS,
    zaxpbyvEVTTest,
    ::testing::Combine(
        ::testing::Values('n' // n: use x, c: use conj(x)
#ifdef TEST_BLIS_TYPED
                          ,
                          'c' // this option is BLIS-api specific.
#endif
                          ),
        ::testing::Values(gtint_t(5)), // m, size of vector to enter NUS loop directly.
        ::testing::Values(gtint_t(3)), // stride size for x
        ::testing::Values(gtint_t(-4)), // stride size for y
        ::testing::Values(gtint_t(0)), // indices to set exception values on x
        ::testing::Values(dcomplex{0.0, 0.0}), // exception values to set on x
        ::testing::Values(gtint_t(0)), // indices to set exception values on y
        ::testing::Values(dcomplex{0.0, 0.0}), // exception values to set on y
        ::testing::Values(dcomplex{NaN, 2.3}, dcomplex{Inf, 0.0}, dcomplex{-Inf, NaN}), // alpha
        ::testing::Values(dcomplex{-0.9, NaN}, dcomplex{0.0, -Inf}, dcomplex{NaN, Inf}) // beta
        ),
    ::zaxpbyvEVTVecPrint());
