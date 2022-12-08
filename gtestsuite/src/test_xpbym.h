#ifndef TEST_XPBYM_H
#define TEST_XPBYM_H

#include "blis_test.h"

double libblis_test_ixpbym_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         alpha,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_xpbym( char* sc_str, obj_t* b, num_t dt );

#endif /* TEST_XPBYM_H */