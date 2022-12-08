#ifndef TEST_AXPYV_H
#define TEST_AXPYV_H

#include "blis_test.h"

double libblis_test_iaxpyv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_axpyv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_AXPYV_H */