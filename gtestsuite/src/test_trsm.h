#ifndef TEST_TRSM_H
#define TEST_TRSM_H

#include "blis_test.h"

double libblis_test_itrsm_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         b_orig,
       num_t          dt
     );

double libblis_check_nan_trsm(obj_t* b, num_t dt );


#endif /* TEST_TRSM_H */