#ifndef TEST_COPYV_H
#define TEST_COPYV_H

#include "blis_test.h"

double libblis_test_icopyv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_copyv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_COPYV_H */