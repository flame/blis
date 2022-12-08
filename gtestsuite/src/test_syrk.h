#ifndef TEST_SYRK_H
#define TEST_SYRK_H

#include "blis_test.h"

double libblis_test_isyrk_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig
     );

double libblis_check_nan_syrk(obj_t* b, num_t dt );

#endif /* TEST_SYRK_H */