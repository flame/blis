#ifndef TEST_NORMFM_H
#define TEST_NORMFM_H

#include "blis_test.h"

double libblis_test_inormfm_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         n
     );

double libblis_check_nan_normfm( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_NORMFM_H */