#ifndef TEST_HERK_H
#define TEST_HERK_H

#include "blis_test.h"

double libblis_test_iherk_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig
     );

double libblis_check_nan_herk(obj_t* b, num_t dt );

#endif /* TEST_HERK_H */