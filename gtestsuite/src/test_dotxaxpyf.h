#ifndef TEST_DOTXAXPYF_H
#define TEST_DOTXAXPYF_H

#include "blis_test.h"

double libblis_test_idotxaxpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         at,
       obj_t*         a,
       obj_t*         w,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         z,
       obj_t*         y_orig,
       obj_t*         z_orig
     );

double libblis_check_nan_dotxaxpyf( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_DOTXAXPYF_H */