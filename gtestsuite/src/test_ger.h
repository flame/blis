#ifndef TEST_GER_H
#define TEST_GER_H

#include "blis_test.h"

double libblis_test_iger_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         y,
       obj_t*         a,
       obj_t*         a_orig
     );

double libblis_check_nan_ger( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_GER_H */