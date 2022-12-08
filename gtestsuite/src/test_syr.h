#ifndef TEST_SYR_H
#define TEST_SYR_H

#include "blis_test.h"

double libblis_test_isyr_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         a,
       obj_t*         a_orig
     );

double libblis_check_nan_syr( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_SYR_H */