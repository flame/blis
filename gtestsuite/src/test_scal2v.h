#ifndef TEST_SCAL2V_H
#define TEST_SCAL2V_H

#include "blis_test.h"

double libblis_test_iscal2v_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig
     );

double libblis_check_nan_scal2v( char*  sc_str, obj_t* b, num_t dt );

#endif /* TEST_SCAL2V_H */