#ifndef TEST_AXPYF_H
#define TEST_AXPYF_H

#include "blis_test.h"

double libblis_test_iaxpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_axpyf( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_AXPYF_H */