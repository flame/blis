#ifndef TEST_DOTV_H
#define TEST_DOTV_H

#include "blis_test.h"

double libblis_test_idotv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho
     );

double libblis_check_nan_dotv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_DOTV_H */