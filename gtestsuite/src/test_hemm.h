#ifndef TEST_HEMM_H
#define TEST_HEMM_H

#include "blis_test.h"

double libblis_test_ihemm_check
    (
      test_params_t* params,
      side_t         side,
      obj_t*         alpha,
      obj_t*         a,
      obj_t*         b,
      obj_t*         beta,
      obj_t*         c,
      obj_t*         c_orig
    );

double libblis_check_nan_hemm(obj_t* c, num_t dt );

#endif /* TEST_HEMM_H */