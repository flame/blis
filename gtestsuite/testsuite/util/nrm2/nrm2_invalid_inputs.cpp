#include <gtest/gtest.h>
#include "test_nrm2.h"
#include "common/wrong_inputs_helpers.h"

/**
 * Testing invalid/incorrect input parameters.
 * 
 * That is only negative n for this API. Zero incx and zero n is allowed.
*/
template <typename T>
class nrm2_IIT : public ::testing::Test {};
typedef ::testing::Types<float, double, scomplex, dcomplex> TypeParam;
TYPED_TEST_SUITE(nrm2_IIT, TypeParam);

// Adding namespace to get default parameters from testinghelpers/common/wrong_input_helpers.h.
using namespace testinghelpers::IIT;

TYPED_TEST(nrm2_IIT, negative_n) {
    using T = TypeParam;    
    using RT = typename testinghelpers::type_info<T>::real_type;
    T x = T{-3.7};
    // initialize blis norm with garbage.
    RT blis_norm = -4.2;
    blis_norm = nrm2<T>(-2, &x, INC);

    computediff<RT>(blis_norm, 0.0);
}
