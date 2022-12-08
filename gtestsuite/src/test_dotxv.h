#ifndef TEST_DOTXV_H
#define TEST_DOTXV_H

#include "blis_test.h"

double libblis_test_idotxv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         y,
       obj_t*         beta,
       obj_t*         rho,
       obj_t*         rho_orig
     );

double libblis_check_nan_dotxv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_DOTXV_H */