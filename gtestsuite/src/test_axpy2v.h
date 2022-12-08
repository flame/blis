#ifndef TEST_AXPY2V_H
#define TEST_AXPY2V_H

#include "blis_test.h"

double libblis_test_iaxpy2v_check
     (
       test_params_t* params,
       obj_t*         alphax,
       obj_t*         alphay,
       obj_t*         x,
       obj_t*         y,
       obj_t*         z,
       obj_t*         z_orig
     );

double libblis_check_nan_axpy2v( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_AXPY2V_H */