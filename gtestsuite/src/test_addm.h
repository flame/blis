#ifndef TEST_ADDM_H
#define TEST_ADDM_H

#include "blis_test.h"

double libblis_test_iaddm_check
    (
      test_params_t* params,
      obj_t*         x,
      obj_t*         y,
      obj_t*         y_orig
    );

double libblis_check_nan_addm( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_ADDM_H */