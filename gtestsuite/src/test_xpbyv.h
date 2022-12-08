#ifndef TEST_XPBYV_H
#define TEST_XPBYV_H

#include "blis_test.h"

double libblis_test_ixpbyv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_xpbyv( char* sc_str, obj_t* b, num_t dt );

#endif /* TEST_XPYV_H */