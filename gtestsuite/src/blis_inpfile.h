#ifndef BLIS_INPFILE_H
#define BLIS_INPFILE_H

#include "blis_test.h"

void libblis_read_inpprms
     (
       string         str,
       test_params_t* params,
       test_ops_t*    ops,
       printres_t*    pfr
     );

void libblis_read_inpops
     (
       string         ss,
       test_params_t* params,
       test_ops_t*    ops,
       string         api,
       printres_t*    pfr
     );

void libblis_read_api
     (
       test_ops_t*  ops,
       opid_t       opid,
       dimset_t     dimset,
       unsigned int n_params,
       test_op_t*   op
     );

int libblis_test_read_randv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_randm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_addv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_amaxv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_axpbyv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_axpyv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_copyv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_dotv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_dotxv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_normfv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_scal2v_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_scalv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_setv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_xpbyv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_subv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_axpyf_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_axpy2v_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_dotxf_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_dotaxpyv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_dotxaxpyf_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_addm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_axpym_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_copym_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_normfm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_scal2m_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_scalm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_setm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_subm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_xpbym_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );


int libblis_test_read_gemv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_ger_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_hemv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_her_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_her2_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_symv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_syr_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_syr2_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_trmv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_trsv_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemmt_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_hemm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_herk_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_her2k_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_symm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_syrk_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_syr2k_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_trmm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_trsm_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_u8s8s32os32_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_u8s8s32os8_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_f32f32f32of32_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_u8s8s32os32_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_u8s8s16os16_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_u8s8s16os8_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_bf16bf16f32of32_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

int libblis_test_read_gemm_bf16bf16f32obf16_params
     (
       char*          str,
       test_op_t*     op,
       test_params_t* params,
       printres_t*    pfr
     );

#endif  // BLIS_INPFILE_H

