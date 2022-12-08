#ifndef TEST_GEMM_H
#define TEST_GEMM_H

#include "blis_test.h"

double libblis_test_igemm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       num_t          dt
     );

double libblis_check_nan_gemm(obj_t* c, num_t dt );

#endif /* TEST_GEMM_H */