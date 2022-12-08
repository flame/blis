#ifndef TEST_AMAXV_H
#define TEST_AMAXV_H

#include "blis_test.h"

double libblis_test_iamaxv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         index
     );

double libblis_check_nan_amaxv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_AMAXV_H */