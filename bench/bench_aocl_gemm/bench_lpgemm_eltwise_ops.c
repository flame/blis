/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "bench_lpgemm_helpers.h"

GEN_FILL_ARRAY_FUNC(float)

GEN_FILL_ARRAY_POST_OPS_FUNC(float)

CONVERT_TO_FLOAT(float)

void print_result
     (
       const char* msg,
       int32_t     n_repeats,
       char        transa,
       char        transb,
       dim_t       m,
       dim_t       n,
       dim_t       lda,
       dim_t       ldb,
       double      gflops
     )
{
    printf("%s transa:%c, transb:%c, m: %ld, n: %ld, lda: %ld, ldb: %ld" \
           " Gops: %f, n_repeats: %d\n",
            msg, transa, transb, m, n, lda, ldb, gflops, n_repeats);
}

#define GEN_ELTWISE_OPS_GET_TEMP_ACCUM(A_type,ACCUM_type,LP_SFX) \
ACCUM_type eltwise_ops_get_temp_accum_ ## LP_SFX \
     ( \
       A_type* a, \
       dim_t rs_a, \
       dim_t cs_a, \
       dim_t i, \
       dim_t j \
     ) \
{ \
    float a_float; \
    bfloat16_to_float( *( a + ( i * rs_a ) + ( j * cs_a ) ), &a_float ); \
    return a_float; \
} \

GEN_ELTWISE_OPS_GET_TEMP_ACCUM(bfloat16,float,bf16of32)
GEN_ELTWISE_OPS_GET_TEMP_ACCUM(bfloat16,float,bf16obf16)

#define GEN_ELTWISE_OPS_GET_TEMP_ACCUM_F(A_type,ACCUM_type,LP_SFX) \
ACCUM_type eltwise_ops_get_temp_accum_ ## LP_SFX \
     ( \
       A_type* a, \
       dim_t rs_a, \
       dim_t cs_a, \
       dim_t i, \
       dim_t j \
     ) \
{ \
    float a_float = *( a + ( i * rs_a ) + ( j * cs_a ) ); \
    return a_float; \
} \

GEN_ELTWISE_OPS_GET_TEMP_ACCUM_F(float,float,f32of32)

GEN_GET_BIAS_POST_OP_VAL(float,bf16of32)
GEN_GET_BIAS_POST_OP_VAL_BF16(bf16obf16)
GEN_GET_BIAS_POST_OP_VAL(float,f32of32)

GEN_GELU_TANH_POSTOP_FLOAT(bf16of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16obf16)
GEN_GELU_TANH_POSTOP_FLOAT(f32of32)

GEN_GELU_ERF_POSTOP_FLOAT(bf16of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16obf16)
GEN_GELU_ERF_POSTOP_FLOAT(f32of32)

GEN_SWISH_POSTOP_FLOAT(bf16of32)
GEN_SWISH_POSTOP_FLOAT(bf16obf16)
GEN_SWISH_POSTOP_FLOAT(f32of32)

static inline float eltwise_ops_accuracy_check_downscale_bf16of32
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }

    float zp_float = *( ( float* )( post_op->sum )->zero_point + j_zp );
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                zp_float );
    return out_temp_accum;
}

static inline float eltwise_ops_accuracy_check_downscale_bf16obf16
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }

    float zp_float = 0.0;
    bfloat16_to_float( *( ( bfloat16* )( post_op->sum )->zero_point + j_zp ),
                       &zp_float );
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                zp_float );
    return out_temp_accum;
}

static inline float eltwise_ops_accuracy_check_downscale_f32of32
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
     dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }

    float zp_float = *( ( float* )( post_op->sum )->zero_point + j_zp );
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                zp_float );
    return out_temp_accum;
}

GEN_GET_MATRIX_ADD_POST_OP_VAL(float,float,bf16of32)
GEN_GET_MATRIX_ADD_POST_OP_VAL_BF16(bfloat16,bf16obf16)
GEN_GET_MATRIX_ADD_POST_OP_VAL(float,float,f32of32)

GEN_GET_MATRIX_MUL_POST_OP_VAL(float,float,bf16of32)
GEN_GET_MATRIX_MUL_POST_OP_VAL_BF16(bfloat16,bf16obf16)
GEN_GET_MATRIX_MUL_POST_OP_VAL(float,float,f32of32)

GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(float,float)

#define GEN_ELTWISE_OPS_ACC_CHK_DRV_FUNC(A_type,B_type,ACCUM_type,LP_SFX) \
void eltwise_ops_accuracy_check_driver_ ## LP_SFX \
     ( \
       FILE*   fout, \
       const char stor_order, \
       char    transa, \
       char    transb, \
       dim_t   m, \
       dim_t   n, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       aocl_post_op*  post_op \
     ) \
{ \
    dim_t rs_a, cs_a; \
    if( ( transa == 'n' ) || ( transa == 'N' ) ) \
    { \
        rs_a = lda; \
        cs_a = 1; \
    } \
    else \
    { \
        rs_a = 1; \
        cs_a = lda; \
    } \
    dim_t rs_b, cs_b; \
    if( ( transb == 'n' ) || ( transb == 'N' ) ) \
    { \
        rs_b = ldb; \
        cs_b = 1; \
    } \
    else \
    { \
        rs_b = 1; \
        cs_b = ldb; \
    } \
 \
    if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
    { \
        if( transa == 'n' || transa == 'N') \
        { \
            rs_a = 1; \
            cs_a = lda; \
        } \
        else \
        { \
            rs_a = lda; \
            cs_a = 1; \
        } \
        if( ( transb == 'n' ) || ( transb == 'N' ) ) \
        { \
            rs_b = 1; \
            cs_b = ldb; \
        } \
        else \
        { \
            rs_b = ldb; \
            cs_b = 1; \
        } \
    } \
 \
    for ( dim_t i = 0; i < m; ++i ) \
    { \
        for ( dim_t j = 0; j < n; ++j ) \
        { \
            ACCUM_type temp_accum = 0; \
            B_type out_temp_accum = 0; \
 \
            temp_accum = GEN_FUNC_NAME(eltwise_ops_get_temp_accum_,LP_SFX) \
                ( a, rs_a, cs_a, i, j ); \
\
            if ( post_op != NULL ) \
            { \
                dim_t ele_i = 0; \
                for ( dim_t op_id = 0; op_id < post_op->seq_length; ++op_id ) \
                { \
                    if ( post_op->seq_vector[op_id] == BIAS ) \
                    { \
                        temp_accum += GEN_FUNC_NAME(get_bias_post_op_val_,LP_SFX) \
                                    ( ( post_op->bias )->bias, j ); \
                    } \
                    else if ( post_op->seq_vector[op_id] == ELTWISE ) \
                    { \
                        if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                PRELU ) /* PReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? \
                                temp_accum : \
                                ( temp_accum * \
                                *( ( ACCUM_type* ) ( post_op->eltwise + ele_i )->algo.alpha ) ); \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_TANH ) /* TANH GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_TANH_post_op_,LP_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_ERF ) /* ERF GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_ERF_post_op_,LP_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                SWISH ) /* SiLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(SWISH_post_op_,LP_SFX) \
                                (temp_accum, \
                                 *( ( ACCUM_type* ) \
                                    ( post_op->eltwise + ele_i )->algo.alpha ) );\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                RELU ) /* ReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                CLIP ) /* CLIP*/ \
                        { \
                            temp_accum = \
                                min \
                                ( \
                                  max \
                                  ( \
                                    temp_accum, \
                                    *( ( ACCUM_type* ) \
                                       ( post_op->eltwise + ele_i )->algo.alpha ) \
                                  ), \
                                  *( ( ACCUM_type* ) \
                                     ( post_op->eltwise + ele_i )->algo.beta) \
                                ); \
                            ele_i += 1; \
                        } \
                        else \
                        {} \
                    } \
                    else if ( post_op->seq_vector[op_id] == SCALE ) \
                    { \
                        temp_accum = GEN_FUNC_NAME(eltwise_ops_accuracy_check_downscale_,LP_SFX) \
                            (temp_accum, post_op, j); \
                    } \
                    else if ( post_op->seq_vector[op_id] == MATRIX_ADD ) \
                    { \
                        dim_t rs_m = ( post_op->matrix_add )->ldm; \
                        dim_t cs_m = 1; \
                        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
                        { \
                            cs_m = rs_m; \
                            rs_m = 1; \
                        } \
                        temp_accum += GEN_FUNC_NAME(get_matrix_add_post_op_val_,LP_SFX) \
                                    ( *( ( B_type* )( post_op->matrix_add )->matrix + \
                                           ( i * rs_m ) + ( j * cs_m ) ) ); \
                    } \
                    else if ( post_op->seq_vector[op_id] == MATRIX_MUL ) \
                    { \
                        dim_t rs_m = ( post_op->matrix_mul )->ldm; \
                        dim_t cs_m = 1; \
                        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
                        { \
                            cs_m = rs_m; \
                            rs_m = 1; \
                        } \
                        temp_accum *= GEN_FUNC_NAME(get_matrix_mul_post_op_val_,LP_SFX) \
                                    ( *( ( B_type* )( post_op->matrix_mul )->matrix + \
                                           ( i * rs_m ) + ( j * cs_m ) ) ); \
                    } \
                    else \
                    {} \
                } \
            } \
            /* Need to convert to downscaled type if required.*/ \
            mat_mul_get_output_type_val ## ACCUM_type ## B_type \
            ( \
              &out_temp_accum, &temp_accum \
            ); \
 \
            if ( *( b + ( rs_b * i ) + ( cs_b * j ) ) != out_temp_accum ) \
            { \
                float comp_float, ref_float; \
                GEN_FUNC_NAME(B_type,_to_float)(*( b + ( rs_b * i ) + ( cs_b * j ) ), &comp_float); \
                GEN_FUNC_NAME(B_type,_to_float)(out_temp_accum, &ref_float); \
                if ( fout ) \
                { \
                    fprintf( fout, "%s Failure input m: %ld, n: %ld," \
                                    " lda: %ld, ldb: %ld, computed:%f, ref:%f, diff:%f\n", \
                                    XSTR(LP_SFX), m, n, lda, ldb, comp_float, \
                                    ref_float, comp_float - ref_float); \
                    fflush( fout ); \
                } \
                    printf("failure, m: %ld, n: %ld, computed:%f, ref:%f, diff:%f\n", i, j, \
                            comp_float, ref_float, comp_float-ref_float); \
                goto cleanup_acc; \
            } \
        } \
    } \
cleanup_acc: \
    return; \
} \

GEN_ELTWISE_OPS_ACC_CHK_DRV_FUNC(bfloat16,float,float,bf16of32)
GEN_ELTWISE_OPS_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,float,bf16obf16)
GEN_ELTWISE_OPS_ACC_CHK_DRV_FUNC(float,float,float,f32of32)

#define GEN_ELTWISE_OPS_BENCH_DRV_FUNC(A_type,B_type,LP_SFX) \
void eltwise_ops_bench_driver_ ## LP_SFX \
     ( \
       char    stor_order, \
       char    transa, \
       char    transb, \
       int32_t n_repeats, \
       dim_t   m, \
       dim_t   n, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       aocl_post_op*  post_op \
     ) \
{ \
    double   dtime;                 \
    double   dtime_save = DBL_MAX;  \
\
    for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
    { \
        dtime = bli_clock();            \
 \
        GEN_FUNC_NAME(aocl_gemm_eltwise_ops_,LP_SFX) \
        ( \
          stor_order, transa, transb, \
          m, n, \
          a, lda, \
          b, ldb, \
          post_op \
        ); \
 \
        dtime_save = bli_clock_min_diff( dtime_save, dtime ); \
 \
    } \
    double gflops = ( m * n ) / ( dtime_save * 1.0e9 ); \
 \
    print_result( XSTR(LP_SFX), n_repeats, transa, transb, m, n, lda, ldb, gflops); \
} \

GEN_ELTWISE_OPS_BENCH_DRV_FUNC(bfloat16,float,bf16of32)
GEN_ELTWISE_OPS_BENCH_DRV_FUNC(bfloat16,bfloat16,bf16obf16)
GEN_ELTWISE_OPS_BENCH_DRV_FUNC(float,float,f32of32)

#define GEN_ELTWISE_OPS_POST_OPS_CREATOR(C_DSCALE_type,C_type,DSCALE_type,BLAS_SFX) \
static inline aocl_post_op* lpgemm_create_post_ops_struct_ ## BLAS_SFX \
     ( \
       dim_t m, \
       dim_t n, \
       char* post_ops_str, \
       char  stor_order \
     ) \
{ \
    if ( ( ( post_ops_str == NULL ) || \
           ( strcmp( post_ops_str, "none" ) == 0 ) ) && \
         ( global_dscale_out == 'n' ) ) \
    { \
        return NULL; \
    } \
 \
    aocl_post_op* post_ops = NULL; \
    post_ops = ( aocl_post_op* ) malloc( sizeof( aocl_post_op ) ); \
 \
    if ( post_ops == NULL ) \
    { \
        return NULL; \
    } \
 \
    /* Only supporting 8 post ops at max for now.*/ \
    dim_t max_post_ops_seq_length = 8; \
    post_ops->seq_vector = ( AOCL_POST_OP_TYPE* ) \
                            malloc \
                            ( \
                              max_post_ops_seq_length * \
                              sizeof( AOCL_POST_OP_TYPE ) \
                            ); \
 \
    if ( post_ops->seq_vector == NULL ) \
    { \
        goto err_handler; \
    } \
 \
    /* Parse post ops list.*/ \
    dim_t cur_op_index = 0; \
    /* Ensure the buffers that use NULL check in deinit code is properly set to NULL.*/ \
    post_ops->eltwise = NULL; \
 \
    /* Bench limitation: can only support 1 bias, but LPGEMM can support
     * multiple scale post-ops. */ \
    post_ops->bias = NULL; \
    post_ops->bias = malloc( sizeof( aocl_post_op_bias ) ); \
    if ( post_ops->bias == NULL ) \
    { \
        goto err_handler; \
    } \
    ( post_ops->bias )->bias = NULL; \
 \
    /* Bench limitation: can only support 1 scale, but LPGEMM can support
     * multiple scale post-ops. */ \
    post_ops->sum = NULL; \
    post_ops->sum = malloc( sizeof( aocl_post_op_sum ) ); \
    if ( post_ops->sum == NULL ) \
    { \
        goto err_handler; \
    } \
    ( post_ops->sum )->scale_factor = NULL; \
    ( post_ops->sum )->buff = NULL; \
    ( post_ops->sum )->zero_point = NULL; \
    ( post_ops->sum )->scale_factor_len = 0; \
    ( post_ops->sum )->zero_point_len = 0; \
 \
    /* Bench limitation: can only support 1 matrix add, but LPGEMM can support
     * multiple matrix add post-ops. */ \
    post_ops->matrix_add = NULL; \
    post_ops->matrix_add = malloc( sizeof( aocl_post_op_matrix_add ) ); \
    if ( post_ops->matrix_add == NULL ) \
    { \
        goto err_handler; \
    } \
    ( post_ops->matrix_add )->matrix = NULL; \
    ( post_ops->matrix_add )->ldm = 0; \
\
    /* Bench limitation: can only support 1 matrix mul, but LPGEMM can support
     * multiple matrix mul post-ops. */ \
    post_ops->matrix_mul = NULL; \
    post_ops->matrix_mul = malloc( sizeof( aocl_post_op_matrix_mul ) ); \
    if ( post_ops->matrix_mul == NULL ) \
    { \
        goto err_handler; \
    } \
    ( post_ops->matrix_mul )->matrix = NULL; \
    ( post_ops->matrix_mul )->ldm = 0; \
 \
    bool is_bias = FALSE; \
    bool is_relu = FALSE; \
    bool is_param_relu = FALSE; \
    bool is_gelu_tanh = FALSE; \
    bool is_gelu_erf = FALSE; \
    bool is_swish = FALSE; \
    bool is_clip = FALSE; \
    bool is_scalar_scale = FALSE; \
    bool is_scalar_zp = FALSE; \
    bool is_matrix_add = FALSE; \
    bool is_matrix_mul = FALSE; \
    dim_t activator_idx = 0; \
    dim_t clip_idx = 0; \
 \
    /* Post-Ops string parser. */ \
    num_eltwise = 0; /* Global variable, zero out for definied behavior. */\
    if ( strcmp( post_ops_str, "none" ) != 0 ) \
    { \
        char* ops_tok = strtok(post_ops_str, ", =" ); \
 \
        /* Ensure only one activator is used as an eltwise post-op.*/ \
        bool is_activator_set = FALSE; \
        while ( ops_tok ) \
        { \
            str_tolower( ops_tok ); \
            if ( strcmp( ops_tok, "bias" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = BIAS; \
                is_bias = TRUE; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "relu" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_relu = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "prelu" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_param_relu = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "swish" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_swish = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "gelu_tanh" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_gelu_tanh = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "gelu_erf" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_gelu_erf = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "clip" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_clip = TRUE; \
                num_eltwise += 1; \
                clip_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "scale" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    is_scalar_scale = TRUE; \
                } \
            } \
            else if ( strcmp( ops_tok, "zp" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    is_scalar_zp = TRUE; \
                } \
            } \
            else if ( strcmp( ops_tok, "matrix_add" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = MATRIX_ADD; \
                is_matrix_add = TRUE; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "matrix_mul" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = MATRIX_MUL; \
                is_matrix_mul = TRUE; \
                cur_op_index++; \
            } \
 \
            ops_tok = strtok( NULL, ", =" ); \
        } \
    } \
 \
    if ( is_bias == TRUE ) \
    { \
        /* Allocate bias buffer, return early if alloc fails.*/ \
        ( post_ops->bias )->bias = malloc( n * sizeof( C_type ) ); \
        if ( ( post_ops->bias )->bias == NULL ) \
        { \
            goto err_handler; \
        } \
        if ( global_dscale_out == 'y' ) \
        { \
            GEN_FUNC_NAME(fill_array_post_ops_,C_DSCALE_type)( ( post_ops->bias )->bias, n ); \
        } \
        else \
        { \
            GEN_FUNC_NAME(fill_array_post_ops_,C_type)( ( post_ops->bias )->bias, n ); \
        } \
    } \
 \
    if ( num_eltwise > 0 ) \
    { \
        if ( num_eltwise > 1 ) \
        { \
            if ( activator_idx < clip_idx ) \
            { \
                activator_idx = 0; \
                clip_idx = 1; \
            } \
            else \
            { \
                activator_idx = 1; \
                clip_idx = 0; \
            } \
        } \
        else \
        { \
           activator_idx = 0; \
           clip_idx = 0; \
        } \
 \
        post_ops->eltwise = malloc( num_eltwise * sizeof( aocl_post_op_eltwise ) ); \
        if ( post_ops->eltwise == NULL ) \
        { \
            goto err_handler; \
        } \
 \
        /* Only one of relu, prelu, swish, gelu_tanh, gelu_erf allowed as
         * an activator. */ \
        if ( is_relu == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = RELU; \
        } \
        else if ( is_param_relu == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( C_type ) ); \
            if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
            { \
                goto err_handler; \
            } \
            *( ( C_type* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( C_type )6; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = PRELU; \
        } \
        if ( is_swish == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( C_type ) ); \
            if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
            { \
                goto err_handler; \
            } \
            *( ( C_type* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( C_type )2; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = SWISH; \
        } \
        else if ( is_gelu_tanh == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = GELU_TANH; \
        } \
        else if ( is_gelu_erf == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = GELU_ERF; \
        } \
        if ( is_clip == TRUE ) \
        { \
            ( post_ops->eltwise + clip_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + clip_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + clip_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + clip_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + clip_idx )->algo.alpha = malloc( sizeof( DSCALE_type ) ); \
            if ( ( post_ops->eltwise + clip_idx )->algo.alpha == NULL ) \
            { \
                goto err_handler; \
            } \
            ( post_ops->eltwise + clip_idx )->algo.beta = malloc( sizeof( DSCALE_type ) ); \
            if ( ( post_ops->eltwise + clip_idx )->algo.beta == NULL ) \
            { \
                goto err_handler; \
            } \
            *( ( DSCALE_type* ) ( post_ops->eltwise + clip_idx )->algo.alpha ) = ( DSCALE_type ) ( -64 ); \
            *( ( DSCALE_type* ) ( post_ops->eltwise + clip_idx )->algo.beta ) = ( DSCALE_type ) ( 23 ); \
            ( post_ops->eltwise + clip_idx )->algo.algo_type = CLIP; \
        } \
    } \
 \
    if ( global_dscale_out == 'y' ) \
    { \
        post_ops->seq_vector[cur_op_index] = SCALE; \
        cur_op_index++; \
 \
        ( post_ops->sum )->is_power_of_2 = FALSE; \
        if ( global_dscale_out == 'y' ) \
        { \
            dim_t n_scale = n; \
            if ( is_scalar_scale == TRUE ) \
            { \
                n_scale = 1; \
            } \
 \
            dim_t n_zp = n; \
            if ( is_scalar_zp == TRUE ) \
            { \
                n_zp = 1; \
            } \
 \
            /* Allocate scale buffer, return early if alloc fails.*/ \
            ( post_ops->sum )->scale_factor = malloc( n_scale * sizeof( DSCALE_type ) ); \
            if ( ( post_ops->sum )->scale_factor == NULL ) \
            { \
                goto err_handler; \
            } \
            ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( C_DSCALE_type ) ); \
            if ( ( post_ops->sum )->zero_point == NULL ) \
            { \
                goto err_handler; \
            } \
 \
            /* Fill scale factor and zero points.*/ \
            DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )( post_ops->sum )->scale_factor; \
            for ( dim_t i = 0; i < n_scale; ++i ) \
            { \
                temp_dscale_ptr[i] = ( ( DSCALE_type )1 )/ ( ( DSCALE_type )1000 ); \
            } \
            ( post_ops->sum )->scale_factor_len = n_scale; \
 \
            C_DSCALE_type* temp_dzero_point_ptr = ( C_DSCALE_type* )( post_ops->sum )->zero_point; \
            GEN_FUNC_NAME(fill_array_,C_DSCALE_type)( temp_dzero_point_ptr, n_zp ); \
            ( post_ops->sum )->zero_point_len = n_zp; \
        } \
    } \
 \
    if ( is_matrix_add == TRUE ) \
    { \
        /* Allocate bias buffer, return early if alloc fails.*/ \
        dim_t ele_dsize = 0; \
        if ( global_dscale_out == 'y' ) \
        { \
            ele_dsize = sizeof( C_DSCALE_type ); \
        } \
        else \
        { \
            ele_dsize = sizeof( C_type ); \
        } \
        ( post_ops->matrix_add )->matrix = malloc( m * n * ele_dsize ); \
        if ( ( post_ops->matrix_add )->matrix == NULL ) \
        { \
            goto err_handler; \
        } \
        if ( global_dscale_out == 'y' ) \
        { \
            GEN_FUNC_NAME(fill_array_,C_DSCALE_type)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
        } \
        else \
        { \
            GEN_FUNC_NAME(fill_array_,C_type)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
        } \
        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
        { \
            ( post_ops->matrix_add )->ldm = m; \
        } \
        else \
        { \
            ( post_ops->matrix_add )->ldm = n; \
        } \
    } \
 \
    if ( is_matrix_mul == TRUE ) \
    { \
        /* Allocate bias buffer, return early if alloc fails.*/ \
        dim_t ele_dsize = 0; \
        if ( global_dscale_out == 'y' ) \
        { \
            ele_dsize = sizeof( C_DSCALE_type ); \
        } \
        else \
        { \
            ele_dsize = sizeof( C_type ); \
        } \
        ( post_ops->matrix_mul )->matrix = malloc( m * n * ele_dsize ); \
        if ( ( post_ops->matrix_mul )->matrix == NULL ) \
        { \
            goto err_handler; \
        } \
        if ( global_dscale_out == 'y' ) \
        { \
            GEN_FUNC_NAME(fill_array_,C_DSCALE_type)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
        } \
        else \
        { \
            GEN_FUNC_NAME(fill_array_,C_type)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
        } \
        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
        { \
            ( post_ops->matrix_mul )->ldm = m; \
        } \
        else \
        { \
            ( post_ops->matrix_mul )->ldm = n; \
        } \
    } \
 \
    post_ops->seq_length = cur_op_index; \
 \
    post_ops->pre_ops = NULL; \
 \
    return post_ops; \
 \
    err_handler: \
    lpgemm_destroy_post_ops_struct( post_ops ); \
    return NULL; \
} \

GEN_ELTWISE_OPS_POST_OPS_CREATOR(bfloat16,float,float,bf16of32)
GEN_ELTWISE_OPS_POST_OPS_CREATOR(bfloat16,bfloat16,float,bf16obf16)
GEN_ELTWISE_OPS_POST_OPS_CREATOR(float,float,float,f32of32)

#define GEN_ELTWISE_OPS_BENCH_MAIN_FUNC(A_type, B_type, LP_SFX) \
void eltwise_ops_bench_main_ ## LP_SFX \
     ( \
       FILE*   fout, \
       char    stor_order, \
       char    transa, \
       char    transb, \
       int32_t m, \
       int32_t n, \
       int32_t stride_a, \
       int32_t stride_b, \
       char*   post_ops_str \
     ) \
{ \
    int32_t n_repeats = bli_max( 30, bli_min( ( 3e10 / ( ( int64_t )m * n ) ), 1000 ) ); \
    if ( global_n_repeat > 0 ) \
    { \
        n_repeats = global_n_repeat; \
    } \
 \
    int32_t size_A = 0; \
    int32_t size_B = 0; \
    if( ( stor_order == 'r' ) || ( stor_order == 'R' ) ) \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? m * stride_a : n * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? m * stride_b : n * stride_b; \
    } \
    else \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? n * stride_a : m * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? n * stride_b : m * stride_b; \
    } \
 \
    A_type* a = ( A_type* ) lpgemm_malloc( sizeof( A_type ) * size_A ); \
    GEN_FUNC_NAME(fill_array_,A_type)(a, size_A ); \
 \
    B_type* b = ( B_type* ) lpgemm_malloc( sizeof( B_type ) * size_B ); \
    memset( ( void* ) b, 0, sizeof( B_type ) * size_B ); \
 \
    if ( bench_mode == 'a' ) \
    { \
        n_repeats = 1; \
    } \
 \
    aocl_post_op* post_op = NULL; \
    if ( ( ( post_ops_str != NULL ) && \
           ( strcmp( post_ops_str, "none" ) != 0 ) ) || \
         ( global_dscale_out == 'y' ) ) \
    { \
        post_op = GEN_FUNC_NAME(lpgemm_create_post_ops_struct_,LP_SFX)( m, n, post_ops_str, stor_order ); \
        if ( post_op == NULL ) \
        { \
            printf(" post op struct allocation failure, returning.\n"); \
            return; \
        } \
    } \
 \
    GEN_FUNC_NAME(eltwise_ops_bench_driver_,LP_SFX) \
    ( \
      stor_order, transa, transb, n_repeats, \
      m, n, \
      a, stride_a, \
      b, stride_b, \
      post_op \
    ); \
 \
    if ( bench_mode == 'a' ) \
    { \
        printf(" Running accuracy check.\n"); \
        GEN_FUNC_NAME(eltwise_ops_accuracy_check_driver_,LP_SFX) \
        ( \
          fout, stor_order, transa, transb, \
          m, n,\
          a, stride_a, \
          b, stride_b, \
          post_op \
        ); \
    } \
 \
    lpgemm_destroy_post_ops_struct( post_op ); \
 \
    lpgemm_free( a ); \
    lpgemm_free( b ); \
} \

GEN_ELTWISE_OPS_BENCH_MAIN_FUNC(bfloat16,float,bf16of32)
GEN_ELTWISE_OPS_BENCH_MAIN_FUNC(bfloat16,bfloat16,bf16obf16)
GEN_ELTWISE_OPS_BENCH_MAIN_FUNC(float,float,f32of32)

int main( int argc, char** argv )
{
    FILE* fin  = NULL;
    if ( argc < 5 )
    {
        printf
        (
          "Usage: ./bench_lpgemm_eltwise_ops -i input.txt -m mode < -n 100 -o op1,op2 >\n" \
          "--Mode is either a or p.\n" \
          "\ta is used for accuracy testing.\n" \
          "\tp is used for performance benchmarking.\n" \
          "--n_repeats can be set optionally using -n arg.\n" \
          "--Post ops can be executed optionaly by providing a coma separated\n" \
          "  list of post-ops after -o arg. Following post-ops are supported:\n" \
          "    1. bias\n" \
          "    2. 4 activators\n" \
          "      a. relu\n" \
          "      b. prelu\n" \
          "      c. gelu_tanh\n" \
          "      d. gelu_erf\n" \
          "    3.clip\n" \
          "  Atleast one post-op needs to be specified if the -o arg is used.\n" \
          "  eg: -o gelu_tanh; -o bias,relu ; -o clip,prelu,bias.\n" \
          "  It is to be noted only one activator can be used at a time.\n" \
          "  If more than one activator is used, only the first activator is\n" \
          "  applied and the other activators are ignored.\n" \
          "  Example: ./bench_lpgemm_eltwise_ops -m a -n 2 -o bias,relu -i input.txt\n" \
        );
        exit( 1 );
    }

    char* file_name = NULL;

#define ELTWISE_OPS_TYPE_STR_LEN 24
    char eltwise_ops_type_str[ELTWISE_OPS_TYPE_STR_LEN];

#define POST_OPS_STR_LEN 104
    char post_ops_str[POST_OPS_STR_LEN];
    char post_ops_str_dest[POST_OPS_STR_LEN]; //Strtok is used to parse, need to maintain a copy.

#define OPS_INPUT_STR_LEN 128
    char ops_input_str[OPS_INPUT_STR_LEN];

    // Parse CLI arguments.
     getopt_t state;
     // Initialize the state for running bli_getopt(). Here, 0 is the
     // initial value for opterr, which suppresses error messages.
     bli_getopt_init_state( 0, &state );

     int opt;
     // Process all option arguments until we get a -1, which means we're done.
     while( (opt = bli_getopt( argc, argv, "i:m:n:", &state )) != -1 )
    {
        char opt_ch = ( char )opt;
        switch( opt_ch )
        {
            case 'i':
                    file_name = state.optarg;
                    break;
            case 'm':
                    bench_mode = ( ( ( *state.optarg ) == 'a' ) || ( ( *state.optarg ) == 'p' ) ) ? ( *state.optarg ) : 'p';
                    break;
            case 'n':
                    global_n_repeat = ( atoi( state.optarg ) > 0 ) ? atoi( state.optarg ) : 0;
                    break;
            default:
                    break;
        }
    }

    if ( bench_mode == 'p' )
    {
        printf( "Running bench in performance benchmarking mode.\n" );
    }
    else if ( bench_mode == 'a' )
    {
        printf( "Running bench in accuracy/correctness testing mode.\n" );
    }

    if ( file_name == NULL )
    {
        printf( " File name provided is invalid.\n" );
        exit( 1 );
    }

    fin = fopen( file_name, "r" );
    if (fin == NULL)
    {
        printf( "Error opening the file %s\n", argv[1] );
        exit( 1 );
    }

    FILE* fout = NULL;

    fout = fopen( "lpgemm_eltwise_ops_accuracy_test_failures.txt", "w" );

    char stor_order;
    char transa, transb;
    int32_t m, n;
    int32_t stride_a, stride_b;

    const dim_t len_list_omp_cores_for_testing = 1;
    const dim_t list_omp_cores_for_testing[1] = { 1 };

    dim_t core_index = 0;
    bool can_run = TRUE;
    while ( ( can_run == TRUE ) && ( fseek( fin, 0L, SEEK_SET ) == 0 ) )
    {
        if ( bench_mode == 'p' )
        {
            can_run = FALSE;
        }
        else if ( bench_mode == 'a' )
        {
            // For accuracy testing, we test accuracy using multiple different
            // number of cores. This helps uncover any bugs related to over
            // subscription or varying thread factorizations.
            // Set current number of cores.
#ifdef BLIS_ENABLE_OPENMP
            omp_set_num_threads( list_omp_cores_for_testing[core_index] );
#endif
            printf( "Accuracy test using %ld threads.\n",
                            list_omp_cores_for_testing[core_index] );

            core_index++;
            if ( core_index < len_list_omp_cores_for_testing )
            {
                can_run = TRUE;
            }
            else
            {
                can_run = FALSE;
            }
        }

        // Input format: data_type stor_type pack m n lda ldb
        while ( fscanf( fin, "%c %c %c %d %d %d %d %s\n",
                &stor_order, &transa, &transb, &m, &n,
                &stride_a, &stride_b, ops_input_str ) == 8 )
        {
            char* ops_tok = strtok( ops_input_str, ":" );
            strncpy( eltwise_ops_type_str, ops_tok, ELTWISE_OPS_TYPE_STR_LEN - 1 );
            str_tolower( eltwise_ops_type_str ); \

            ops_tok = strtok( NULL, "" );
            if ( ops_tok != NULL )
            {
                strncpy( post_ops_str, ops_tok, POST_OPS_STR_LEN - 1 );
            }
            else
            {
                strncpy( post_ops_str, "none", POST_OPS_STR_LEN - 1 );
            }

            stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
                            ( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
                            stor_order : 'r';

            if ( ( strcmp( eltwise_ops_type_str, "bf16of32" ) == 0 ) ||
                 ( strcmp( eltwise_ops_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                GEN_FUNC_NAME(eltwise_ops_bench_main_, bf16of32)
                (
                    fout, stor_order, transa, transb,
                    m, n, stride_a, stride_b,
                    post_ops_str_dest
                );
            }
            if ( ( strcmp( eltwise_ops_type_str, "bf16obf16" ) == 0 ) ||
                 ( strcmp( eltwise_ops_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'y';
                GEN_FUNC_NAME(eltwise_ops_bench_main_, bf16obf16)
                (
                    fout, stor_order, transa, transb,
                    m, n, stride_a, stride_b,
                    post_ops_str_dest
                );
            }
            if ( ( strcmp( eltwise_ops_type_str, "f32of32" ) == 0 ) ||
                 ( strcmp( eltwise_ops_type_str, "*" ) == 0 ) )
            {
                strncpy( post_ops_str_dest, post_ops_str, POST_OPS_STR_LEN );
                global_dscale_out = 'n';
                GEN_FUNC_NAME(eltwise_ops_bench_main_, f32of32)
                (
                    fout, stor_order, transa, transb,
                    m, n, stride_a, stride_b,
                    post_ops_str_dest
                );
            }
        }
    }

    if ( fin )
    {
        fclose( fin );
    }
    if ( fout )
    {
        fclose( fout );
    }
    return 0;
}
