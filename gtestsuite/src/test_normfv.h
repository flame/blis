#ifndef TEST_NORMFV_H
#define TEST_NORMFV_H

#include "blis_test.h"

double libblis_test_inormfv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         n
     );

double libblis_check_nan_normfv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_NORMFV_H */