#ifndef TEST_HER2_H
#define TEST_HER2_H

#include "blis_test.h"

double libblis_test_iher2_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         y,
       obj_t*         a,
       obj_t*         a_orig
     );

double libblis_check_nan_her2( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_HER2_H */