#ifndef TEST_DOTAXPYV_H
#define TEST_DOTAXPYV_H

#include "blis_test.h"

double libblis_test_idotaxpyv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         xt,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho_orig,
       obj_t*         z,
       obj_t*         z_orig
     );

double libblis_check_nan_dotaxpyv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_DOTAXPYV_H */