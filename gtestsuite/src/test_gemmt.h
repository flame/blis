#ifndef TEST_GEMMT_H
#define TEST_GEMMT_H

#include "blis_test.h"

double libblis_test_igemmt_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig
     );

template <typename T, typename U>
void libblis_igemv_check
     (
       trans_t transA,
       dim_t   M,
       dim_t   N,
       T*      alpha,
       T*      A,
       dim_t   rsa,
       dim_t   csa,
       T*      X,
       dim_t   incx,
       T*      beta,
       T*      Y,
       dim_t   incy
     );

template <typename T, typename U>
void libblis_icgemv_check
     (
       trans_t transA,
       dim_t   M,
       dim_t   N,
       T*      alpha,
       T*      A,
       dim_t   rsa,
       dim_t   csa,
       bool    conja,
       T*      X,
       dim_t   incx,
       T*      beta,
       T*      Y,
       dim_t   incy,
       bool    conjx
     );

template <typename T, typename U>
void libblis_igemm_check
     (
       dim_t M,
       dim_t N,
       dim_t K,
       T*    alpha,
       T*    A,
       dim_t rsa,
       dim_t csa,
       T*    B,
       dim_t rsb,
       dim_t csb,
       T*    beta,
       T*    C,
       dim_t rsc,
       dim_t csc
     );

template <typename T, typename U>
void libblis_icgemm_check
     (
       dim_t M,
       dim_t N,
       dim_t K,
       T*    alpha,
       T*    A,
       dim_t rsa,
       dim_t csa,
       bool  conja,
       T*    B,
       dim_t rsb,
       dim_t csb,
       bool  conjb,
       T*    beta,
       T*    C,
       dim_t rsc,
       dim_t csc
     );

double libblis_check_nan_gemmt( obj_t* c );

#endif /* TEST_GEMMT_H */