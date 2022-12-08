#ifndef TEST_TRMV_H
#define TEST_TRMV_H

#include "blis_test.h"

double libblis_test_itrmv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         x_orig
     );

double libblis_check_nan_trmv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_TRMV_H */