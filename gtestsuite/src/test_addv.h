#ifndef TEST_ADDV_H
#define TEST_ADDV_H

#include "blis_test.h"

double libblis_test_iaddv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         beta,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_addv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_ADDV_H */