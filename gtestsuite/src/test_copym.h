#ifndef TEST_COPYM_H
#define TEST_COPYM_H

#include "blis_test.h"

double libblis_test_icopym_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_copym( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_COPYM_H */