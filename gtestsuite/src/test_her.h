#ifndef TEST_HER_H
#define TEST_HER_H

#include "blis_test.h"

double libblis_test_iher_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         a,
       obj_t*         a_orig
     );

double libblis_check_nan_her( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_HER_H */