#ifndef TEST_SCALM_H
#define TEST_SCALM_H

#include "blis_test.h"

double libblis_test_iscalm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_scalm( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_SCALM_H */