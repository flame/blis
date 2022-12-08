#ifndef TEST_AXPBYV_H
#define TEST_AXPBYV_H

#include "blis_test.h"

double libblis_test_iaxpbyv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_axpbyv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_AXPBYV_H */