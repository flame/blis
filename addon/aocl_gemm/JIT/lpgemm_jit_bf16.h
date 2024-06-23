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

#ifndef JIT_BF16_H
#define JIT_BF16_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <functional>
#include "blis.h"
#include <xbyak/xbyak.h>

using namespace Xbyak;

class bli_lpgemm_jit: public Xbyak::CodeGenerator
{

private :
    void preamble();
    void postamble();
    void initialize_params( lpgemm_jit_inputs_t* params );
    void reg_init(dim_t m_dim, dim_t n_dim );
    void kernel_unroll( dim_t m_dim, dim_t n_dim );
    void prefetchC( dim_t m_dim, dim_t n_dim );
    void k_fringe_loop( dim_t m_dim, dim_t n_dim );
    void scale_alpha( dim_t m_dim, dim_t n_dim );
    // beta ops
    void bf16_f32_beta_op( dim_t m_dim, dim_t n_dim );
    void f32_f32_beta_op( dim_t m_dim, dim_t n_dim );
    //postops
    void clip_f32( dim_t m_dim, dim_t n_dim );
    void f32_f32_matrix_add( dim_t m_dim, dim_t n_dim );
    void bf16_f32_matrix_add( dim_t m_dim, dim_t n_dim );
    void bias_row_major( dim_t m_dim, dim_t n_dim );
    void bias_col_major( dim_t m_dim, dim_t n_dim );
    void relu( dim_t m_dim, dim_t n_dim );
    void relu_scale( dim_t m_dim, dim_t n_dim );
    void gelu_tanh( dim_t m_dim, dim_t n_dim );
    void POLY_EVAL_6_AVX512();
    void EXPF_AVX512();
    void TANHF_AVX512();
    void GELU_TANH_F32_AVX512_DEF( dim_t reg );
    void POLY_EVAL_HORNER_16_0_AVX512();
    void ERF_AVX512();
    void GELU_ERF_F32_AVX512_DEF( dim_t reg );
    void gelu_erf( dim_t m_dim, dim_t n_dim );
    void SWISH_F32_AVX512_DEF( dim_t reg );
    void swish( dim_t m, dim_t n );

    void apply_post_ops_in_high_reg_pressure
    (
      const dim_t num_post_op_regs,
      std::function< void( dim_t ) > op_fn
    );
    // C store functions
    void cvt_store_f32_bf16_mask( dim_t m_dim, dim_t n_dim );
    void store_f32( dim_t m_dim, dim_t n_dim );

    void post_op_label_lastk_safe_jump_with_next_ptr();
    void post_op_label_lastk_safe_jump();


    dim_t num_elems_per_reg = 64 / sizeof(float);
    dim_t n_rem;
    dim_t num_fma_regs;
    dim_t fma_start_idx = 0;
    dim_t load_start_idx = 0;
    dim_t num_full_loads;
    dim_t num_loads;
    dim_t bcst_start_idx;
    dim_t alpha_reg = fma_start_idx;
    dim_t beta_reg;

    // registers used for gelu_tanh
    const dim_t num_gelu_regs = 9;
    const dim_t const1 = load_start_idx;
    const dim_t const2 = load_start_idx+1;
    const dim_t x = load_start_idx+2;
    const dim_t r = load_start_idx+3;
    const dim_t r2 = load_start_idx+4;
    const dim_t z = load_start_idx+5;
    const dim_t dn = load_start_idx+6;
    const dim_t x_tanh = load_start_idx+7;
    const dim_t q = load_start_idx+8;

        // registers for gelu_erf
    const dim_t num_erf_regs = 5;
    const dim_t x_erf = load_start_idx+4;

    // registers used for swish. Reusing the gelu_tanh registers.
    const dim_t num_swish_regs = 9;

    const dim_t stack_off_ps_a = 8;
    const dim_t stack_off_k_iter_before_prefetch = 16;
    const dim_t stack_off_k_iter_after_prefetch = 24;
    const dim_t stack_off_k_left = 32;
    const dim_t stack_off_alpha = 40;
    const dim_t stack_off_beta = 48;
    const dim_t stack_off_b_ptr = 56;
    const dim_t stack_off_postop = 64;
    const dim_t stack_off_buf_downscale = stack_off_postop +
                                          offsetof( lpgemm_post_op_attr,
                                                    buf_downscale );
    const dim_t stack_off_temp_list = stack_off_postop +
                                      sizeof( lpgemm_post_op );


    const dim_t stack_off_zmm_stack = stack_off_temp_list + 8;
          dim_t zmm_stack_top;

    void store_zmms_in_stack( dim_t reg_start_idx,
                              dim_t num_regs,
                              dim_t stack_off
                            );

    void get_zmms_from_stack( dim_t reg_start_idx,
                              dim_t num_regs,
                              dim_t stack_off
                            );

    float gelu_consts[7] = { 0.044715, 0.797884, -2,  0.5, -1, 2, 1 };
    float gelu_macros[6] = { 1.4426950408889634, 1.2582912E7,
                                   -88.0f,             88.0f,
                                   (float)(1.0/0.0),  -2147483648 };

    float lpgemm_exp[6] = { 1.0000000754895704,  0.6931472254087585,
                                  0.2402210737432219,  0.05550297297702539,
                                  0.009676036358193323, 0.001341000536524434 };

    float erf_consts[4] = { 0.707107, 1.0, 0.5, 3.553f };

    float lpgemm_erf[16] = { 1.1283793786592402,    2.5468861568875563E-5,
                                   0.3756169877289898,    0.004025179163741976,
                                   0.12947984300439994,   0.0412525204794885,
                                   0.03918550001070417,   0.07104542913277255,
                                   0.05717052146749476,   0.025310822854733135,
                                   0.0067305713376882076, 0.0010410692067591445,
                                   6.921588102382636E-5,  4.092409485758739E-6,
                                   1.033131746125426E-6,  5.2927177513236435E-8 };


    const dim_t gelu_consts_off = 0;
    const dim_t gelu_macros_off = gelu_consts_off + sizeof(gelu_consts);
    const dim_t lpgemm_exp_off = gelu_macros_off + sizeof(gelu_macros);
    const dim_t erf_consts_off = lpgemm_exp_off + sizeof(lpgemm_exp);
    const dim_t lpgemm_erf_off = erf_consts_off + sizeof(erf_consts);

    Xbyak::Address get_constant( dim_t table_off, dim_t value_off )
    {
        return ptr[rip + tables + table_off + value_off * 4 ];
    }
    Xbyak::Label tables;

public:
    bli_lpgemm_jit( void* buffer, size_t bufferSize );
    void generate_kernel( lpgemm_jit_inputs_t* params );
    const void (*get_function ()const)( lpgemm_jit_params_t*,
                                        lpgemm_post_op_attr*,
                                        lpgemm_post_op*
                                      );
    const void *get_code ()const;
    dim_t get_size ();

};
#endif
