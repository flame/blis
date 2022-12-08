#ifndef TEST_HER2K_H
#define TEST_HER2K_H

#include "blis_test.h"

double libblis_test_iher2k_check
    (
      test_params_t* params,
      obj_t*         alpha,
      obj_t*         a,
      obj_t*         b,
      obj_t*         beta,
      obj_t*         c,
      obj_t*         c_orig
    );

double libblis_check_nan_her2k(obj_t* c, num_t dt );

#endif /* TEST_HER2K_H */