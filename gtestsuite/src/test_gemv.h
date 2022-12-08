#ifndef TEST_GEMV_H
#define TEST_GEMV_H

#include "blis_test.h"

double libblis_test_igemv_check
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  x,
       obj_t*  beta,
       obj_t*  y,
       obj_t*  y_orig,
       num_t   dt
     );

double libblis_check_nan_gemv( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_GEMV_H */