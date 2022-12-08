#ifndef TEST_SCALV_H
#define TEST_SCALV_H

#include "blis_test.h"

double libblis_test_iscalv_check
     (
       test_params_t* params,
       obj_t*         beta,
       obj_t*         x,
       obj_t*         x_orig
     );

double libblis_check_nan_scalv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_SCALV_H */