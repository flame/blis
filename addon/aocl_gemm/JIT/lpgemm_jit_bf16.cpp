/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "lpgemm_jit_bf16.h"

// push callee-save registers to stack
void bli_lpgemm_jit:: preamble()
{
    push(rbp);
    push(rbx);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
}

// pop the callee-save registers before returning from function.
void bli_lpgemm_jit:: postamble()
{
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbx);
    pop(rbp);
    vzeroupper();
}

void bli_lpgemm_jit:: store_zmms_in_stack( dim_t reg_start_idx,
                                           dim_t num_regs,
                                           dim_t stack_off
                                         )
{
    for( dim_t idx = 0; idx < num_regs; idx++ )
    {
        vmovups( ptr[ rsp + zmm_stack_top + stack_off + idx * 64],
                 Zmm( reg_start_idx + idx ) );
    }
}

void bli_lpgemm_jit:: get_zmms_from_stack( dim_t reg_start_idx,
                                           dim_t num_regs,
                                           dim_t stack_off
                                         )
{
    for( dim_t idx = 0; idx < num_regs; idx++ )
    {
        vmovups( Zmm( reg_start_idx + idx ),
                 ptr[ rsp + zmm_stack_top + stack_off + idx * 64] );
    }
}

//Zero out the registers that will be used for storing accumulated values.
// For a given micro-kernel dimension MRxNR,
// considering a row-major kernel, we need (MR * (NR / num_elems per reg))
// registers to store accumulated values.
void bli_lpgemm_jit:: reg_init( dim_t m_dim, dim_t n_dim )
{
    vxorps( Zmm( fma_start_idx ), Zmm( fma_start_idx ));
    for( dim_t m = fma_start_idx + 1; m < 32; m++ )
    {
        vmovaps( Zmm( m ), Zmm( fma_start_idx ) );
    }
}


// This code replicates the existing bf16 kernel.
// Hence unroll factor is hardcoded to be 2.
// To-DO: Make unroll factor as an configurable parameter.
#ifdef BPREFETCH_JIT
void bli_lpgemm_jit:: kernel_unroll( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;
    dim_t cnt = 0;
    // Broadcast elements of A matrix
    vpbroadcastd( Zmm( bcst_start_idx ), ptr[ rax ] );

    // load elements of B matrix into registers
    for( dim_t n = 0; n < num_full_loads; n++ )
        vmovdqu16( Zmm( load_start_idx + n ), ptr[ rbx + n * 64 ] );

    // In case of last load with fringe part, use mask
    if( n_rem )
        vmovdqu16( Zmm( load_start_idx + num_full_loads )
                        | k3 | T_z, ptr[ rbx + num_full_loads * 64 ] );

    add( rbx, r10 );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        // broadcast elements of A matrix.
        // Using 2 ZMM registers for broadcast.
        if( m < ( m_dim - 1 ) )
        {
            switch ( m + 1 )
            {
            case 1:
            case 4:
            case 2: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r8 * ( m + 1 ) ] );
                    break;
            case 3: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r13 ] );
                    break;
            case 5: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r15 ] );
                    break;
            default:
                break;
            }
        }

        // move to next column
        if( m == ( m_dim - 1 ) ) add( rax, r9 );
        if(cnt < num_full_loads)
        {
            prefetcht0( ptr[ rcx + cnt * 64 ] );
            cnt++;
        }
        // Generate FMA instructions.
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            vdpbf16ps( Zmm( reg_num ), Zmm( bcst_start_idx + m % 2 ),
                       Zmm( load_start_idx + n ) );
        }
    }
    add( rcx, num_full_loads * 64 );
}
#else
void bli_lpgemm_jit:: kernel_unroll( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    // Broadcast elements of A matrix
    vpbroadcastd( Zmm( bcst_start_idx ), ptr[ rax ] );

    // load elements of B matrix into registers
    for( dim_t n = 0; n < num_full_loads; n++ )
        vmovdqu16( Zmm( load_start_idx + n ), ptr[ rbx + n * 64 ] );

    // In case of last load with fringe part, use mask
    if( n_rem )
        vmovdqu16( Zmm( load_start_idx + num_full_loads )
                        | k3 | T_z, ptr[ rbx + num_full_loads * 64 ] );

    add( rbx, r10 );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        // broadcast elements of A matrix.
        // Using 2 ZMM registers for broadcast.
        if( m < ( m_dim - 1 ) )
        {
            switch ( m + 1 )
            {
            case 1:
            case 4:
            case 2: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r8 * ( m + 1 ) ] );
                    break;
            case 3: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r13 ] );
                    break;
            case 5: vpbroadcastd( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r15 ] );
                    break;
            default:
                break;
            }
        }

        // move to next column
        if( m == ( m_dim - 1 ) ) add( rax, r9 );

        // Generate FMA instructions.
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            vdpbf16ps( Zmm( reg_num ), Zmm( bcst_start_idx + m % 2 ),
                       Zmm( load_start_idx + n ) );
        }
    }
}
#endif

void bli_lpgemm_jit:: k_fringe_loop( dim_t m_dim, dim_t n_dim )
{

    dim_t reg_num;

    // Broadcast elements of A matrix
    vpbroadcastw( Zmm( bcst_start_idx ), ptr[ rax ] );

    // load elements of B matrix into registers
    for( dim_t n = 0; n < num_full_loads; n++ )
        vmovdqu16( Zmm( load_start_idx + n ), ptr[ rbx + n * 64 ] );

    // In case of last load with fringe part, use mask
    if( n_rem )
        vmovdqu16( Zmm( load_start_idx + num_full_loads )
                        | k3 | T_z, ptr[ rbx + num_full_loads * 64 ] );


    for( dim_t m = 0; m < m_dim; m++ )
    {
        if( m < ( m_dim - 1 ) )
        {
            // broadcast elements of A matrix.
            // Using 2 ZMM registers for broadcast.
            switch ( m + 1 )
            {
            case 1:
            case 4:
            case 2: vpbroadcastw( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r8 * ( m + 1 ) ] );
                    break;
            case 3: vpbroadcastw( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r13 ] );
                    break;
            case 5: vpbroadcastw( Zmm( bcst_start_idx + ( m + 1 ) % 2 ),
                                  ptr[ rax + r15 ] );
                    break;
            default:
                break;
            }
        }

        // Generate FMA instructions.
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            vdpbf16ps( Zmm( reg_num ), Zmm( bcst_start_idx + m % 2 ),
                       Zmm( load_start_idx + n ) );
        }

    }
}

// Generate required number of mul instructions for scaling with alpha.
void bli_lpgemm_jit:: scale_alpha( dim_t m_dim, dim_t n_dim )
{
    for( dim_t reg_num = fma_start_idx; reg_num < 32; reg_num++ )
        vmulps( Zmm( reg_num ), Zmm( alpha_reg ), Zmm( reg_num ) );
}


// Scale C by beta and store when beta is a generic value.
void bli_lpgemm_jit:: f32_f32_beta_op( dim_t m_dim, dim_t n_dim)
{
    dim_t reg_num;
    for( dim_t m = 0; m < m_dim; m++ )
    {
        if( m > 0 ) add( rcx, rdi );

        for( dim_t n = 0; n < num_full_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            vmovups( Zmm( load_start_idx + n ) , ptr[ rcx + n * 64 ] );

            vfmadd231ps( Zmm( reg_num ), Zmm( load_start_idx + n ),
                         Zmm( beta_reg ) );
        }

        // Use mask in case of n_fringe.
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;

            vmovups( Zmm( load_start_idx + num_full_loads ) | k4 | T_z,
                     ptr[ rcx + num_full_loads * 64 ] );

            vfmadd231ps( Zmm( reg_num ),
                         Zmm( load_start_idx + num_full_loads ),
                         Zmm( beta_reg ) );
        }
    }
}

void bli_lpgemm_jit:: bf16_f32_beta_op( dim_t m_dim, dim_t n_dim )
{

    dim_t reg_num;
    mov( rcx, ptr[ rsp + stack_off_buf_downscale ] );
    mov( rax, ptr[ rsp + stack_off_postop + offsetof( lpgemm_post_op_attr,
                                                      rs_c_downscale ) ] );



    // rs_c_downscale *= sizeof(bfloat16)
    lea( rax, ptr[ rax * 2 ] );
    mov( rsi, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(bfloat16) )
    imul( rsi, rax );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(bfloat16) )
    //       + post_op_c_j * sizeof(bfloat16)
    lea( rsi, ptr[ rsi + rbx * 2 ] );

    add( rcx, rsi );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_full_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            // convert from 16 bit elements to 32 bit elements
            vpmovsxwd( Zmm( load_start_idx + n ), ptr[ rcx + n * 32 ] );

            // Shift left by 16 bits
            vpslld( Zmm( load_start_idx + n ), Zmm( load_start_idx + n ),
                     0x10 );

            // fma with beta
            vfmadd231ps( Zmm( reg_num ), Zmm( beta_reg ),
                         Zmm( load_start_idx + n ) );
        }
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;

            // load the bf16 elements from the downscale buffer using mask.
            vmovdqu16( Ymm( load_start_idx + num_full_loads ) | k4 | T_z,
                       ptr[rcx + num_full_loads * 32 ] );

            // convert from 16 bit elements to 32 bit elements
            vpmovsxwd( Zmm( load_start_idx + num_full_loads ),
                       Ymm( load_start_idx + num_full_loads ) );

            // Shift left by 16 bits
            vpslld( Zmm( load_start_idx + num_full_loads ),
                     Zmm( load_start_idx + num_full_loads ), 0x10 );

            // fma with beta
            vfmadd231ps( Zmm( reg_num ), Zmm( beta_reg ),
                         Zmm( load_start_idx + num_full_loads ) );
        }

        // move to next row
        add( rcx, rax );
    }

}

void bli_lpgemm_jit:: clip_f32( dim_t m_dim, dim_t n_dim )
{
    dim_t min_reg = load_start_idx;
    dim_t max_reg = bcst_start_idx;

    // min reg
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args2 ) ] );
    vbroadcastss( Zmm( min_reg ), ptr[ rax ] );

    // max reg
    mov( rbx, ptr[ rdx + offsetof( lpgemm_post_op, op_args3 ) ] );
    vbroadcastss( Zmm( max_reg ), ptr[ rbx ] );

    for( dim_t m = fma_start_idx; m < 32; m++ )
    {
        vmaxps( Zmm( m ), Zmm( m ), Zmm( min_reg ) );
        vminps( Zmm( m ), Zmm( m ), Zmm( max_reg ) );
    }
}

void bli_lpgemm_jit:: bf16_f32_matrix_add( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    // rcx =  matrix ptr
    mov( rcx, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );

    // rax = ldm
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args3 ) ] );
    mov( rax, ptr[ rax ] );

    // ldm *= sizeof(bfloat16)
    lea( rax, ptr[ rax * 2 ] );

    mov( rsi, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(bfloat16) )
    imul( rsi, rax );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(bfloat16) )
    //       + post_op_c_j * sizeof(bfloat16)
    lea( rsi, ptr[ rsi + rbx * 2 ] );

    add( rcx, rsi );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_full_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;

            // convert from 16 bit elements to 32 bit elements
            vpmovsxwd( Zmm( load_start_idx + n ), ptr[ rcx + n*32 ] );

            // Shift left by 16 bits
            vpslld( Zmm( load_start_idx + n ), Zmm( load_start_idx + n ),
                     0x10 );

            vaddps( Zmm( reg_num ), Zmm( reg_num ),
                    Zmm( load_start_idx + n ) );

        }
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;

            // load the bf16 elements from the downscale buffer using mask.
            vmovdqu16( Ymm( load_start_idx + num_full_loads ) | k4 | T_z,
                       ptr[rcx + num_full_loads * 32 ] );

            // convert from 16 bit elements to 32 bit elements
            vpmovsxwd( Zmm( load_start_idx + num_full_loads ),
                       Ymm( load_start_idx + num_full_loads ) );

            // Shift left by 16 bits
            vpslld( Zmm(load_start_idx + num_full_loads ),
                     Zmm( load_start_idx + num_full_loads ), 0x10 );

            vaddps( Zmm( reg_num ), Zmm( reg_num ),
                    Zmm( load_start_idx + num_full_loads ) );
        }

        // move to next row
        add( rcx, rax );
    }
}


void bli_lpgemm_jit:: f32_f32_matrix_add( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    // rcx =  matrix ptr
    mov( rcx, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );
    // rax = ldm
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args3 ) ] );
    mov( rax, ptr[ rax ] );

    // ldm *= sizeof(float)
    lea( rax, ptr[ rax * 4 ] );

    mov( rsi, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(float) )
    imul( rsi, rax );

    // rsi = post_op_c_i * ( rs_c_downscale * sizeof(float) )
    //       + post_op_c_j * sizeof(float)
    lea( rsi, ptr[ rsi + rbx * 4] );

    add( rcx, rsi );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_full_loads; n++)
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vmovups(Zmm( load_start_idx + n ), ptr[ rcx + n * 64 ] );
            vaddps( Zmm( reg_num ), Zmm( reg_num ),
                    Zmm( load_start_idx + n ) );
        }
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;
            vmovups( Zmm( load_start_idx + num_full_loads ) | k4 | T_z,
                     ptr[ rcx + num_full_loads * 64 ] );
            vaddps( Zmm( reg_num ), Zmm( reg_num ),
                    Zmm( load_start_idx + num_full_loads ) );
        }

        // move to next row
        add( rcx, rax );
    }
}
void bli_lpgemm_jit:: bias_row_major( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );

    mov( rcx, ptr[ rdx + offsetof( lpgemm_post_op, stor_type ) ] );
    cmp( rcx, 4 );
    je( "BIAS_BF16_ROW_MAJOR", T_NEAR );

    // postops_c_j *= sizeof(float)
    lea( rbx, ptr[ rbx * 4 ] );
    add( rax, rbx );
    for( dim_t n = 0; n < num_full_loads; n++ )
    {
        vmovups( Zmm( load_start_idx + n ), ptr[ rax + n * 64 ] );
    }
    if( n_rem )
    {
        vmovups( Zmm( load_start_idx + num_full_loads ) | k4,
                 ptr[ rax + num_full_loads * 64 ] );
    }
    jmp( "POST_BIAS_BF16_ROW_MAJOR", T_NEAR );

    L( "BIAS_BF16_ROW_MAJOR" );
    // postops_c_j *= sizeof(bfloat16)
    lea( rbx, ptr[ rbx * 2 ] );
    add( rax, rbx );
    for( dim_t n = 0; n < num_full_loads; n++ )
    {
        // convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( load_start_idx + n ), ptr[ rax + n * 32 ] );

        // Shift left by 16 bits
        vpslld( Zmm( load_start_idx + n ), Zmm( load_start_idx + n ), 0x10 );
    }
    if( n_rem )
    {
        // load the bf16 elements from the downscale buffer using mask.
        vmovdqu16( Ymm( load_start_idx + num_full_loads ) | k4 | T_z,
                   ptr[rax + num_full_loads * 32 ] );

        // convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( load_start_idx + num_full_loads ),
                   Ymm( load_start_idx + num_full_loads ) );

        // Shift left by 16 bits
        vpslld( Zmm( load_start_idx + num_full_loads ),
                 Zmm( load_start_idx + num_full_loads ), 0x10 );
    }
    L( "POST_BIAS_BF16_ROW_MAJOR" );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vaddps( Zmm( reg_num ), Zmm( reg_num ),
                    Zmm( load_start_idx + n ) );
        }
    }
}

void bli_lpgemm_jit:: bias_col_major( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );

    mov( rcx, ptr[ rdx + offsetof( lpgemm_post_op, stor_type ) ] );
    cmp( rcx, 4 );
    je( "BIAS_BF16_COL_MAJOR", T_NEAR );

    // postops_c_i *= sizeof(float)
    lea( rbx, ptr[ rbx * 4 ] );
    add( rax, rbx );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        vbroadcastss( Zmm( alpha_reg ), ptr[ rax + m *  4 ] );
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vaddps( Zmm( reg_num ), Zmm( reg_num ), Zmm( alpha_reg ) );
        }
    }
    jmp( "POST_BIAS_BF16_COL_MAJOR", T_NEAR );

    L( "BIAS_BF16_COL_MAJOR" );
    // postops_c_i *= sizeof(bfloat16)
    lea( rbx, ptr[ rbx * 2 ] );
    add( rax, rbx );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        vpbroadcastw( Zmm( alpha_reg ), ptr[ rax + m * 4 ] );

        // convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( alpha_reg ), Ymm( alpha_reg ) );

        // Shift left by 16 bits
        vpslld( Zmm( alpha_reg ), Zmm( alpha_reg ), 0x10 );

        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vaddps( Zmm( reg_num ), Zmm( reg_num ), Zmm( alpha_reg ) );
        }
    }
    L( "POST_BIAS_BF16_COL_MAJOR" );
}

void bli_lpgemm_jit:: relu( dim_t m_dim, dim_t n_dim )
{
    dim_t scratch_reg = bcst_start_idx;

    vpxorq(Zmm( scratch_reg ), Zmm( scratch_reg ), Zmm( scratch_reg ) );

    for( dim_t m = fma_start_idx; m < 32; m++ )
    {
        vmaxps( Zmm( m ), Zmm( m ), Zmm( scratch_reg ) );
    }
}

void bli_lpgemm_jit:: relu_scale( dim_t m_dim, dim_t n_dim )
{
    dim_t zero_reg = load_start_idx;
    dim_t scale_factor = bcst_start_idx;

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args2 ) ] );
    vbroadcastss( Zmm( scale_factor ), ptr[ rax ] );
    vpxorq( Zmm( zero_reg ), Zmm( zero_reg ), Zmm( zero_reg ) );

    for( dim_t m = fma_start_idx; m < 32; m++ )
    {
        vcmpps( k5, Zmm( m ), Zmm( zero_reg ), 0x02 );
        vmulps( Zmm( m ) | k5, Zmm( m ), Zmm( scale_factor ) );
    }
}

void bli_lpgemm_jit:: downscale_row_major( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;
    dim_t sf_reg = load_start_idx;
    dim_t zp_reg = bcst_start_idx;

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, scale_factor ) ] );

    mov( rbx, ptr[ rdx + offsetof( lpgemm_post_op, scale_factor_len ) ] );
    cmp( rbx, 1 );
    je( "DOWNSCALE_ROW_MAJOR_SF_EQ1");

    //If scale_factor_length > 1 load each element in the register.
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );  //
    // post_op_c_j *= sizeof( float )
    lea( rbx, ptr[ rbx * 4 ] );
    add( rax, rbx );
    for( dim_t n = 0; n <  num_full_loads; n++ )
    {
        vmovups( Zmm( sf_reg + n ), ptr[ rax + n * 64 ] );
    }
    if( n_rem )
    {
        vmovups( Zmm( sf_reg + num_full_loads ) | k4,
                 ptr[ rax + num_full_loads * 64 ] );
    }
    jmp( "DOWNSCALE_ZERO_POINT" );

    L( "DOWNSCALE_ROW_MAJOR_SF_EQ1");
    //Broadcast the scale_factor value. When scale-factor length == 1,
    //even though different registers are used to hold the scalar value
    //all those registers would contain the same value.
    for( dim_t n = 0; n < num_full_loads; n++ )
        vbroadcastss( Zmm( sf_reg + n ), ptr[ rax ] );

    if(n_rem)
    {
        vbroadcastss( Zmm( sf_reg + num_full_loads ), ptr[ rax ] );
    }

    L( "DOWNSCALE_ZERO_POINT" );
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );

    mov( rbx, ptr[ rdx + offsetof( lpgemm_post_op, op_args3 ) ] );
    mov( rbx, ptr[ rbx ] );
    cmp( rbx, 1 );
    je( "DOWNSCALE_ROW_MAJOR_ZP_EQ1" );

    // load post_op_c_j and multiply with sizeof(bfloat16)
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );
    // post_op_c_j *= sizeof( bfloat16 )
    lea( rbx, ptr[ rbx * 2 ] );
    add( rax, rbx );
    //If zero_point > 1
    for( dim_t n = 0; n < num_full_loads; n++ )
    {
        //Convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( zp_reg + n ), ptr[ rax + n * 32 ] );

        //Shift left by 16 bits
        vpslld( Zmm( zp_reg + n ), Zmm( zp_reg + n ), 0x10 );
    }
    if(n_rem)
    {
        // load the bf16 elements from the downscale buffer using mask.
        vmovdqu16( Ymm( zp_reg + num_full_loads ) | k4 | T_z,
                   ptr[rax + num_full_loads * 32 ] );

        // convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( zp_reg + num_full_loads ),
                   Ymm( zp_reg + num_full_loads ) );

        // Shift left by 16 bits
        vpslld( Zmm( zp_reg + num_full_loads ),
                 Zmm( zp_reg + num_full_loads ), 0x10 );
    }
    jmp( "POST_SCALAR_ZP" );

    L( "DOWNSCALE_ROW_MAJOR_ZP_EQ1" );
    for( dim_t n = 0; n < num_full_loads; n++ )
    {
        vpbroadcastw( Ymm( zp_reg + n ), ptr[ rax ] );
        //Convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( zp_reg + n ), Ymm( zp_reg + n ) );

        //Shift left by 16 bits
        vpslld( Zmm( zp_reg + n ), Zmm( zp_reg + n ), 0x10 );
    }

    if(n_rem)
    {
        // load the bf16 elements from the downscale buffer using mask.
        vpbroadcastw( Ymm( zp_reg + num_full_loads ) | k4 | T_z,
                   ptr[ rax ] );

        // convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( zp_reg + num_full_loads ),
                   Ymm( zp_reg + num_full_loads ) );

        // Shift left by 16 bits
        vpslld( Zmm( zp_reg + num_full_loads ),
                 Zmm( zp_reg + num_full_loads ), 0x10 );
    }

    L( "POST_SCALAR_ZP" );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vmulps( Zmm( reg_num ), Zmm( reg_num ), Zmm( sf_reg + n ) );
            vaddps( Zmm( reg_num ), Zmm( reg_num ), Zmm( zp_reg + n ) );
        }
    }
}

void bli_lpgemm_jit:: downscale_col_major( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, scale_factor ) ] );

    mov( rbx, ptr[ rdx + offsetof( lpgemm_post_op, scale_factor_len ) ] );
    cmp( rbx, 1 );
    je( "DOWNSCALE_COL_MAJOR_SF_EQ1" );
    // If scale_factor_length > 1, broadcast each element in the register.
    // In order to save on the registers the scale factor values are
    // broadcast in all m_dim registers and caluculate the scale values first.
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    // post_op_c_i *= sizeof( float )
    lea( rbx, ptr[ rbx * 4 ] );
    add( rax, rbx );
    for( dim_t m = 0; m < m_dim; m++ )
        vbroadcastss( Zmm( load_start_idx + m ), ptr[ rax + m * 4 ] );

    jmp( "DOWNSCALE_COL_SCALE_FACTOR" );

    L( "DOWNSCALE_COL_MAJOR_SF_EQ1");
    //Broadcast the scale_factor value. When scale-factor length == 1,
    //even though different registers are used to hold the scalar value
    //all those registers would contain the same value.
     for( dim_t m = 0; m < m_dim; m++ )
        vbroadcastss( Zmm( load_start_idx + m ), ptr[ rax ] );

    L( "DOWNSCALE_COL_SCALE_FACTOR" );
    //Calculate the scale factor value with the broadcasted scale factor values.
    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vmulps( Zmm( reg_num ), Zmm( reg_num ), Zmm( load_start_idx + m ) );
        }
    }

    //Zero-Point
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args1 ) ] );

    mov( rbx, ptr[ rdx + offsetof( lpgemm_post_op, op_args3 ) ] );
    mov( rbx, ptr[ rbx ] );
    cmp( rbx, 1 );
    je( "DOWNSCALE_COL_MAJOR_ZP_EQ1" );

    //If zp_length > 1 broadcast each element in the register.
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    // post_op_c_i *= sizeof( bfloat16 )
    lea( rbx, ptr[ rbx * 2 ] );
    add( rax, rbx );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        //broadcast the bf16 elements from the downscale buffer using mask.
        vpbroadcastw( Ymm( load_start_idx + m ), ptr[ rax + m * 2 ] );

        //Convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( load_start_idx + m ), Ymm( load_start_idx + m ) );

        //Shift left by 16 bits
        vpslld( Zmm( load_start_idx + m ), Zmm( load_start_idx + m ), 0x10 );
    }

    jmp( "DOWNSCALE_COL_ZERO_POINT" );

    L( "DOWNSCALE_COL_MAJOR_ZP_EQ1" );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        //Broadcast the bf16 elements from the downscale buffer
        vpbroadcastw( Ymm( load_start_idx + m ), ptr[ rax ] );

        //Convert from 16 bit elements to 32 bit elements
        vpmovsxwd( Zmm( load_start_idx + m ), Ymm( load_start_idx + m ) );

        //Shift left by 16 bits
        vpslld( Zmm( load_start_idx + m ), Zmm( load_start_idx + m ), 0x10 );
    }

    L( "DOWNSCALE_COL_ZERO_POINT" );
    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vaddps( Zmm( reg_num ), Zmm( reg_num ), Zmm( load_start_idx + m ) );
        }
    }
}

void bli_lpgemm_jit::apply_post_ops_in_high_reg_pressure
     (
       const dim_t num_post_op_regs,
       std::function< void( dim_t ) > op_fn
     )
{
    dim_t num_push_regs = num_post_op_regs - fma_start_idx ;

    // If number of registers required to compute pots op is more than
    // registers available, then push some accum registers to stack
    // and use them to compute gelu.
    store_zmms_in_stack( fma_start_idx, num_push_regs, 0 );

    dim_t post_op_start = num_push_regs > 0 ? fma_start_idx + num_push_regs
                                         : fma_start_idx;

    // operate on non-pushed regs
    for( dim_t reg = post_op_start; reg < 32; reg++ )
    {
        op_fn( reg );
    }

    // Push num_push_regs number of registers from last to stack and
    // replace them with the items that were pushed earlier
    // and compute on them.
    store_zmms_in_stack( 32 - num_push_regs, num_push_regs,
                         num_push_regs * 64 );
    get_zmms_from_stack( 32 - num_push_regs, num_push_regs, 0);

    for( dim_t reg = 0; reg < num_push_regs; reg++ )
    {
        op_fn( 32 - num_push_regs + reg );
    }

    for( dim_t reg = 0; reg < num_push_regs; reg++ )
        vmovups( Zmm( fma_start_idx + reg ),
                 Zmm( 32 - num_push_regs + reg ) );

    get_zmms_from_stack( 32 - num_push_regs, num_push_regs,
                         num_push_regs * 64 );
}

//r2 and z, q are scratch regs
//r will be passed in and out of parent function.
void bli_lpgemm_jit:: POLY_EVAL_6_AVX512( )
{
    vmulps( Zmm( r2 ), Zmm( r ), Zmm( r ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_exp_off, 3) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_exp_off, 2) );

    vmovups( Zmm( q ), Zmm( const2 ) );
    vfmadd231ps( Zmm( q ), Zmm( const1 ), Zmm( r ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_exp_off, 1) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_exp_off, 0) );

    vmovups( Zmm( z ), Zmm( const2 ) );
    vfmadd231ps( Zmm( z ), Zmm( const1 ), Zmm( r ) );

    vfmadd231ps( Zmm( z ), Zmm( r2 ), Zmm( q ) );

    vmulps(Zmm( r2 ), Zmm( r2 ), Zmm( r2 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_exp_off, 5) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_exp_off, 4) );

    vfmadd231ps( Zmm( const2 ), Zmm( const1 ), Zmm( r ) );

    vfmadd231ps( Zmm( z ), Zmm( const2 ), Zmm( r2 ) );
    vmovups(Zmm( r ), Zmm( z ) );
}

// z, r, dn is a scratch register
// takes 'x' as input and returns 'q' to the parent
void bli_lpgemm_jit:: EXPF_AVX512()
{
    vbroadcastss( Zmm( const1 ), get_constant(gelu_macros_off, 0) );

    vmulps( Zmm( z ), Zmm( x ), Zmm(const1 ) );

    vbroadcastss( Zmm( const2 ), get_constant(gelu_macros_off, 1) );

    vaddps( Zmm( dn ), Zmm( z ), Zmm( const2 ) );

    vsubps( Zmm( r ), Zmm( dn ), Zmm( const2 ) );
    vsubps( Zmm( r ), Zmm( z ), Zmm( r ) );

    POLY_EVAL_6_AVX512();

    vpslld( Zmm( dn ), Zmm( dn ), 0x17 );

    vpaddd( Zmm( q ), Zmm( r ), Zmm( dn ) );

    vpxorq( Zmm( const2 ), Zmm( const2 ), Zmm( const2 ) );

    vpbroadcastd( Zmm( const1 ), get_constant(gelu_macros_off, 2) );

    vcmpps( k5, Zmm( const1 ), Zmm( x ), 0x06 );

    vpandd( Zmm( q ) | k5, Zmm( q ), Zmm( const2 ) );

    vbroadcastss( Zmm( const1 ), get_constant(gelu_macros_off, 3) );

    vcmpps( k5, Zmm( const1 ), Zmm( x ), 0x06 );

    vbroadcastss( Zmm( x ), get_constant(gelu_macros_off, 4) );

    vpxord( Zmm( x ) | k5, Zmm( q ), Zmm( const2 ) );
    vmovups(Zmm( q ), Zmm( x ) );
}

// uses z, dn, r as scratch regs
// passes r to child macro and gets q
// takes x_tanh as input and gives back x_tanh
void bli_lpgemm_jit:: TANHF_AVX512()
{
    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 2) );

    mov( ebx, 0x7FFFFFFF );
    vpbroadcastd( Zmm( const2 ), ebx );
    vpandd( Zmm( x ), Zmm( x_tanh ), Zmm( const2 ) );

    vmulps( Zmm( x ), Zmm( x ), Zmm( const1 ) );

    EXPF_AVX512();

    mov( eax, -1 );
    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 4) );

    vaddps( Zmm( z ), Zmm( q ), Zmm( const1 ) );

    vbroadcastss( Zmm( const2 ), get_constant(gelu_consts_off, 5) );

    vaddps( Zmm( r ), Zmm( z ), Zmm( const2 ) );

    vdivps( Zmm( z ), Zmm( z ), Zmm( r ) );

    vmulps( Zmm( z ), Zmm( z ), Zmm( const1 ) );

    mov( eax, -2147483648 );
    vpbroadcastd( Zmm( const1 ), eax );

    vpandd(Zmm( q ), Zmm( x_tanh ), Zmm( const1 ) );

    vpxord( Zmm( x_tanh ), Zmm( q ), Zmm( z ) );
}

void bli_lpgemm_jit:: GELU_TANH_F32_AVX512_DEF(dim_t reg )
{
    vmulps( Zmm( r2 ), Zmm( reg ), Zmm( reg ) );
    vmulps( Zmm( r2 ), Zmm( r2 ), Zmm( reg ) );

    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 0) );
    vmovups( Zmm( r ), Zmm( reg ) );
    vfmadd231ps( Zmm( r ), Zmm( r2 ), Zmm( const1 ) );

    vbroadcastss( Zmm( const2 ), get_constant(gelu_consts_off, 1) );
    vmulps( Zmm( x_tanh ), Zmm( r ), Zmm( const2 ) );

    TANHF_AVX512();

    vbroadcastss( Zmm( const2 ), get_constant(gelu_consts_off, 6) );
    vaddps( Zmm( x_tanh ), Zmm( x_tanh ), Zmm( const2 ) );
    vmulps( Zmm( x_tanh ), Zmm( x_tanh ), Zmm( reg ) );

    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 3) );
    vmulps( Zmm( reg ), Zmm( x_tanh ), Zmm( const1 ) );
}

void bli_lpgemm_jit:: gelu_tanh( dim_t m_dim, dim_t n_dim )
{
    apply_post_ops_in_high_reg_pressure
    (
      num_gelu_regs,
      std::bind
      (
        &bli_lpgemm_jit::GELU_TANH_F32_AVX512_DEF,
        this,
        std::placeholders::_1
      )
    );
}

void bli_lpgemm_jit:: POLY_EVAL_HORNER_16_0_AVX512()
{
    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 15) );
    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 14) );

    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 13) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 12) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 11) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 10) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 9) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 8) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 7 ) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 6) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 5) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 4) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 3) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 2) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vbroadcastss( Zmm( const1 ), get_constant(lpgemm_erf_off, 1) );
    vfmadd231ps( Zmm( const1 ), Zmm( r ), Zmm( const2 ) );

    vbroadcastss( Zmm( const2 ), get_constant(lpgemm_erf_off, 0) );
    vfmadd231ps( Zmm( const2 ), Zmm( r ), Zmm( const1 ) );

    vmulps( Zmm( x ), Zmm( const2 ), Zmm( r ) );
}

void bli_lpgemm_jit:: ERF_AVX512()
{
    mov( eax, 0x7FFFFFFF );
    vpbroadcastd( Zmm( const2 ), eax );
    vpandd( Zmm( r ), Zmm( x_erf ), Zmm( const2 ) );

    POLY_EVAL_HORNER_16_0_AVX512();

    vbroadcastss( Zmm( const1 ), get_constant(erf_consts_off, 1) );

    vbroadcastss( Zmm( const2 ), get_constant(erf_consts_off, 3) );

    vcmpps( k5, Zmm( const2 ), Zmm( r ), 0x06 );

    vpxorq( Zmm( const2 ), Zmm( const2 ), Zmm( const2 ) );

    vpxord( Zmm( const1 ) | k5, Zmm( x ), Zmm( const2 ) );
    vmovups( Zmm( x ), Zmm( const1 ) );


    vbroadcastss( Zmm( const1 ), get_constant(erf_consts_off, 1) );

    vcmpps( k5, Zmm( const1 ), Zmm( x ), 0x06 );

    vpxord( Zmm( const1 ) | k5, Zmm( x ), Zmm( const2 ) );

    mov( eax, ~(0x7FFFFFFF));
    vpbroadcastd( Zmm( const2 ), eax );

    vpandd( Zmm( x_erf ), Zmm( x_erf ), Zmm( const2 ) );

    vpord( Zmm( x_erf ), Zmm( x_erf ), Zmm( const1 ) );
}

void bli_lpgemm_jit:: GELU_ERF_F32_AVX512_DEF( dim_t reg )
{
    vbroadcastss( Zmm( const1 ), get_constant(erf_consts_off, 0) );
    vmulps( Zmm( x_erf ), Zmm( reg ), Zmm( const1 ) );

    ERF_AVX512();

    vbroadcastss( Zmm( const2 ), get_constant(erf_consts_off, 1) );
    vaddps( Zmm( x_erf ), Zmm( x_erf ), Zmm( const2 ) );

    vmulps( Zmm( x_erf ), Zmm( x_erf ), Zmm( reg ) );
    vbroadcastss( Zmm( const2 ), get_constant(erf_consts_off, 2) );
    vmulps( Zmm( reg ), Zmm( x_erf ), Zmm( const2 ) );

}

void bli_lpgemm_jit:: gelu_erf( dim_t m_dim, dim_t n_dim )
{
    apply_post_ops_in_high_reg_pressure
    (
      num_gelu_regs,
      std::bind
      (
        &bli_lpgemm_jit::GELU_ERF_F32_AVX512_DEF,
        this,
        std::placeholders::_1
      )
    );
}

void bli_lpgemm_jit::SWISH_F32_AVX512_DEF( dim_t reg )
{
    vpxorq( Zmm( x ), Zmm( x ), Zmm( x ) );
    vfnmadd231ps( Zmm( x ), Zmm( reg ), Zmm( x_tanh ) );

    // Input reg x and output reg q.
    EXPF_AVX512();

    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 6) );
    vaddps( Zmm( q ), Zmm( q ), Zmm( const1 ) );
    vdivps( Zmm( reg ), Zmm( reg ), Zmm( q ) );
}

void bli_lpgemm_jit::swish( dim_t m_dim, dim_t n_dim )
{
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args2 ) ] );
    vbroadcastss( Zmm( x_tanh ), ptr[ rax ] );

    apply_post_ops_in_high_reg_pressure
    (
      num_gelu_regs,
      std::bind
      (
        &bli_lpgemm_jit::SWISH_F32_AVX512_DEF,
        this,
        std::placeholders::_1
      )
    );
}

void bli_lpgemm_jit::TANHF_AVX512_DEF( dim_t reg )
{
    vpxorq( Zmm( x ), Zmm( x ), Zmm( x ) );
    vmovups( Zmm( x_tanh ), Zmm( reg ) );
    TANHF_AVX512();
    vmovups( Zmm( reg ), Zmm( x_tanh ) );
}

void bli_lpgemm_jit::tanh( dim_t m_dim, dim_t n_dim )
{
    apply_post_ops_in_high_reg_pressure
    (
      num_gelu_regs,
      std::bind
      (
        &bli_lpgemm_jit::TANHF_AVX512_DEF,
        this,
        std::placeholders::_1
      )
    );
}

void bli_lpgemm_jit::SIGMOID_AVX512_DEF( dim_t reg )
{
    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 4) );
    vmulps( Zmm( x ), Zmm( const1 ), Zmm( reg ) );

    //Input is x, output is q
    EXPF_AVX512();

    vbroadcastss( Zmm( const1 ), get_constant(gelu_consts_off, 6) );
    vaddps( Zmm( q ), Zmm( q ), Zmm( const1 ) );
    vdivps( Zmm( reg ), Zmm( q ), Zmm( const1 ) );
}

void bli_lpgemm_jit::sigmoid( dim_t m_dim, dim_t n_dim )
{
    apply_post_ops_in_high_reg_pressure
    (
      num_gelu_regs,
      std::bind
      (
        &bli_lpgemm_jit::SIGMOID_AVX512_DEF,
        this,
        std::placeholders::_1
      )
    );
}

void bli_lpgemm_jit:: store_f32( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;
    for( dim_t m = 0; m < m_dim; m++ )
    {
        if( m > 0 ) add( rcx, rdi );

        for( dim_t n = 0; n < num_full_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            vmovups( ptr[ rcx + n * 64 ],  Zmm( reg_num ) );
        }

        // Use mask in case of n_fringe.
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;
            vmovups( ptr[ rcx + num_full_loads * 64 ] | k4, Zmm( reg_num ) );
        }
    }
}
void bli_lpgemm_jit:: cvt_store_f32_bf16_mask( dim_t m_dim, dim_t n_dim )
{
    dim_t reg_num;

    mov( rcx, ptr[ rsp + stack_off_buf_downscale ] );
    mov( rax, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, rs_c_downscale ) ] );

    // rs_c_downscale *= sizeof(bfloat16)
    lea( rax, ptr[rax * 2 ] );
    mov( rsi, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    mov( rbx, ptr[ rsp + stack_off_postop +
                   offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );

    imul( rsi, rax );
    lea( rsi, ptr[ rsi + rbx * 2 ] );
    add( rcx, rsi );

    for( dim_t m = 0; m < m_dim; m++ )
    {
        for( dim_t n = 0; n < num_full_loads; n++ )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + n;
            // convert from 32 bit elements to 16 bit elements
            vcvtneps2bf16( Ymm( reg_num ), Zmm( reg_num ) );
            vmovdqu16( ptr[ rcx + n * 32 ], Ymm( reg_num ) );
        }
        if( n_rem )
        {
            reg_num = fma_start_idx + ( m * num_loads ) + num_full_loads;
            // convert from 32 bit elements to 16 bit elements
            vcvtneps2bf16( Ymm( reg_num ), Zmm( reg_num ) );
            vmovdqu16( ptr[ rcx + num_full_loads * 32 ] | k4, Ymm( reg_num ) );
       }
        // move to next row
        add( rcx, rax );
    }
}

void bli_lpgemm_jit::initialize_params( lpgemm_jit_inputs_t* params )
{
    // params needed in kernel
    // a(r14, rax), b(rbx), c(r12, rcx) podim_ters. To be stored in regs
    // rs_a(r8), cs_a(r9), rs_b(r10), rs_c(rdi).
    // alpha(rax), beta(rbx) values. To be pushed to stack
    // m_iter(r11), ps_a(rax) values. ps_a to be pushed to stack.
    // k_iter(rsi), k_left(rsi) value. To be pushed to stack.

    // load values from params struct to registers and stack
    if( params->m_loop )
    {
        // move address of a
        mov( r14, ptr[ rdi + offsetof( lpgemm_jit_params_t, a ) ] );
        mov( r11, ptr[ rdi + offsetof( lpgemm_jit_params_t, m_iter ) ] );
    }
    else
    {
        mov( rax, ptr[ rdi + offsetof(lpgemm_jit_params_t, a ) ] );
    }

    if( params->generate_mask )
    {
        // This mask will be used to load/store bf16 elements
        kmovd( k3, ptr[ rdi + offsetof( lpgemm_jit_params_t, mask16 ) ] );
        // This mask will be used to load/store f32 elements
        kmovw( k4, ptr[ rdi + offsetof(lpgemm_jit_params_t, mask32 ) ] );
    }

    mov( r12, ptr[ rdi + offsetof( lpgemm_jit_params_t, c ) ] );
    mov( r8,  ptr[ rdi + offsetof( lpgemm_jit_params_t, rs_a ) ] );
    mov( r9,  ptr[ rdi + offsetof( lpgemm_jit_params_t, cs_a ) ] );
    mov( r10, ptr [rdi + offsetof( lpgemm_jit_params_t, rs_b ) ] );


    // Push all the params that will be required in later stages
    // of kernel to stack.
    // Pusing in order ps_a2, k_iter, k_left, alpha, beta, b
    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, ps_a2 ) ] );
    mov( ptr[ rsp + stack_off_ps_a ], rbx);

    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t,
                                   k_iter_before_prefetch ) ] );
    mov( ptr[ rsp + stack_off_k_iter_before_prefetch ], rbx );

    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t,
                                   k_iter_after_prefetch ) ] );
    mov( ptr[ rsp + stack_off_k_iter_after_prefetch ], rbx );

    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, k_left ) ] );
    mov( ptr[ rsp + stack_off_k_left ], rbx );

#ifdef BPREFETCH_JIT
    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, bprefetch_dist ) ] );
    mov( ptr[ rsp + stack_off_bprefetch_dist ], rbx );
#endif
    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, alpha ) ] );
    mov( ptr[ rsp + stack_off_alpha ], rbx );

    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, beta ) ] );
    mov( ptr[ rsp + stack_off_beta ], rbx );

    mov( rbx, ptr[ rdi + offsetof( lpgemm_jit_params_t, b ) ] );
    mov( ptr[ rsp + stack_off_b_ptr ], rbx );

    // once all the params that will be required in
    // later stages of kernel are pushed to stack,
    // move rs_c dim_to rdi.
    mov( rdi, ptr[ rdi + offsetof( lpgemm_jit_params_t, rs_c ) ] );


    // push all members of lpgemm_post_op_attr struct to stack.
    // Since this will be passed as 2nd arg to the function, it will be in rsi

    mov( rbx, ptr[ rsi + offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, post_op_c_i ) ], rbx );

    mov( rcx, ptr[ rsi + offsetof( lpgemm_post_op_attr, post_op_c_j ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, post_op_c_j ) ], rcx );

    mov( rbx, ptr[ rsi + offsetof( lpgemm_post_op_attr, rs_c_downscale ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, rs_c_downscale)], rbx );

    mov( rcx, ptr[ rsi + offsetof( lpgemm_post_op_attr, cs_c_downscale ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, cs_c_downscale)], rcx );

    mov( rbx, ptr[ rsi + offsetof(lpgemm_post_op_attr, buf_downscale ) ] );
    mov( ptr[ rsp + stack_off_buf_downscale ], rbx );

    mov( rcx, ptr[ rsi + offsetof( lpgemm_post_op_attr, is_first_k ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, is_first_k ) ], rcx );

    mov( rbx, ptr[ rsi + offsetof(lpgemm_post_op_attr, is_last_k ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, is_last_k ) ], rbx );

    mov( rcx, ptr[ rsi + offsetof( lpgemm_post_op_attr, c_stor_type ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, c_stor_type ) ], rcx );

    mov( rbx, ptr[ rsi + offsetof(lpgemm_post_op_attr, b_sum_offset)]);
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, b_sum_offset )] , rbx );

    mov( rcx, ptr[ rsi + offsetof( lpgemm_post_op_attr, b_col_sum_vec ) ] );
    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, b_col_sum_vec ) ], rcx );

    mov( rbx, ptr[ rsi +
                  offsetof( lpgemm_post_op_attr, b_col_sum_vec_s16 ) ] );

    mov( ptr[ rsp + stack_off_postop +
              offsetof( lpgemm_post_op_attr, b_col_sum_vec_s16 ) ], rbx );

    // Storing the address to the head node of post-op list in stack
    // It needs to be restored after every loop of m_iter
    mov( ptr[ rsp + stack_off_temp_list ], rdx );

    // initialize top of zmm stack
    zmm_stack_top = stack_off_zmm_stack;
}

void bli_lpgemm_jit:: prefetchC( dim_t m_dim, dim_t n_dim )
{
    for( dim_t m = 0; m < m_dim; m++ )
    {
        if( m > 0 ) add( rcx, rdi );
        for( dim_t n = 0; n < num_loads; n++ )
        {
            prefetcht1( ptr[ rcx + n * 64 ] );
        }
    }
}

void bli_lpgemm_jit:: post_op_label_lastk_safe_jump_with_next_ptr()
{
    mov( rdx, ptr[rdx+offsetof( lpgemm_post_op, next ) ] );
    post_op_label_lastk_safe_jump();
}
void bli_lpgemm_jit:: post_op_label_lastk_safe_jump()
{
    // check if post_ops_list_temp != NULL
    cmp( rdx, 0 );
    je( "POST_OPS_6x64_DISABLE", T_NEAR );

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_code ) ] );
    cmp( rax, POST_OPS_DISABLE );
    je( "POST_OPS_6x64_DISABLE", T_NEAR );
    cmp( rax, POST_OPS_BIAS ) ;
    je( "POST_OPS_BIAS_6x64", T_NEAR );
    cmp( rax, POST_OPS_RELU );
    je( "POST_OPS_RELU_6x64", T_NEAR );
    cmp( rax, POST_OPS_RELU_SCALE );
    je( "POST_OPS_RELU_SCALE_6x64", T_NEAR );
    cmp( rax, POST_OPS_GELU_TANH );
    je( "POST_OPS_GELU_TANH_6x64", T_NEAR );
    cmp( rax, POST_OPS_GELU_ERF );
    je( "POST_OPS_GELU_ERF_6x64", T_NEAR );
    cmp( rax, POST_OPS_CLIP );
    je( "POST_OPS_CLIP_6x64", T_NEAR );
    cmp( rax, POST_OPS_DOWNSCALE );
    je( "POST_OPS_DOWNSCALE_6x64", T_NEAR );
    cmp( rax, POST_OPS_MATRIX_ADD );
    je( "POST_OPS_MATRIX_ADD_6x64", T_NEAR );
    cmp( rax, POST_OPS_SWISH );
    je( "POST_OPS_SWISH_6x64", T_NEAR );
    cmp( rax, POST_OPS_TANH );
    je( "POST_OPS_TANH_6x64", T_NEAR );
    cmp( rax, POST_OPS_SIGMOID );
    je( "POST_OPS_SIGMOID_6x64", T_NEAR );
}

// Constructor
bli_lpgemm_jit:: bli_lpgemm_jit( void* buffer, size_t bufferSize )
               : CodeGenerator( bufferSize, buffer )
{
    protect( buffer, bufferSize, PROTECT_RWE );
}

// Main kernel function body
void bli_lpgemm_jit::generate_kernel( lpgemm_jit_inputs_t* params )
{

    dim_t m_dim = params->MR;
    dim_t n_dim = params->NR;

    // In kernel-function pointer array, kernels to handle n < 16
    // are stored at col-index 0. Hacking n_dim to some value 0 < value < 16
    // so masked instructions are generated.
    // This will be removed when we support on-the-fly generation of kernels.
    if( n_dim == 0 )
    {
        n_dim = 2;
        params->generate_mask = TRUE;
    }


    n_rem = n_dim % NUM_F32_ELEMS_PER_ZMM;

    // Number of loads that doesn't require mask
    num_full_loads = ( n_dim / num_elems_per_reg );

    // Number of loads in total =  full loads + mask load (if required)
    num_loads = ( num_full_loads ) + ( n_rem > 0 ? 1 : 0 );

    // Total number of registers to store accumulated values.
    num_fma_regs = m_dim * num_loads;

    // calculating start index for accumulation registers.
    // If the kernel requires 'x' number of accumulation regs, we use the
    // last 'x' ZMMs available on certain architecture.
    // 31 is hardcoded here since we only support AVX-512 as of now,
    // This needs to be made as a configurable parameter later.
    fma_start_idx = 31 - num_fma_regs + 1;

    // If a kernel requires x registers for loads, we always use the
    // first 'x' ZMM registers available for loads.
    // And the immediate registers next to load regs are used for broadcast.
    bcst_start_idx = load_start_idx + num_loads;

    // While scaling the accumulated registers with beta,
    // load regs will be used to load C matrix,
    // Hence using broadcast register to store beta value.
    beta_reg = bcst_start_idx;

    preamble();
    // add some spack in stack to store params
    sub( rsp, 512 );
    // Initialize all the paramters required for execution of kernel.
    // load some values to registers and push the rest of them to stack.
    initialize_params( params );

/* register usage:
    r14, rax - podim_ter for A matrix
    r8  - rs_a
    r9  - cs_a
    r13 - 3 * rs_a
    r15 - 5 * rs_a
    rbx - podim_ter to B matrix, beta
    r10 -  rs_b
    r12, rcx - podim_ter for C matrix
    rdi - rs_c
    r11 - m_iter
    rsi - k_iter, k_left
    rax - ps_a2, alpha
*/


    lea( rdi, ptr[ rdi * 4 ] );          // rs_c *= sizeof(float) => rs_c *= 4

    lea( r8, ptr[ r8 * 2 ] );            // rs_a *= sizeof(dt) => rs_a *= 2
    lea( r9, ptr[ r9 * 2 ] );            // cs_a *= sizeof(dt) => cs_a *= 2
    if ( m_dim >= 4)
        lea( r13, ptr[r8 + r8 * 2 ] );   // r13 = 3 * rs_a
    if( m_dim >= 6 )
        lea( r15, ptr[r8 + r8 * 4 ] );   // r15 = 5 * rs_a

    lea( r10, ptr[ r10 * 2 ] );          // rs_b *= sizeof(dt) => rs_b *= 2


    mov( rcx, r12 );

    if( params->m_loop )
    {

        L( "BLOOP6X64I" );
        mov( rax, r14 );                 // reset rax to current upanel of a.
    }


    mov( rbx, ptr[ rsp + stack_off_b_ptr ] );  // move address of b


    // Zero all the registers that will be used for accumulation.
    reg_init( m_dim, n_dim );
    prefetcht0( ptr[ rbx ] );

#ifdef BPREFETCH_JIT
    // load k_iter
    mov( rsi, ptr[ rsp + stack_off_k_iter_before_prefetch ] );
    test( rsi, rsi );
    je( "BCONSIDKLEFT", T_NEAR );

    mov( rcx, ptr[ rsp + stack_off_bprefetch_dist ] );
    add( rcx, rbx );

    L( "BLOOPKITER" );
    // Main k-unroll loop with 4 kernel unroll
    kernel_unroll( m_dim, n_dim );

    kernel_unroll( m_dim, n_dim );

    kernel_unroll( m_dim, n_dim );

    kernel_unroll( m_dim, n_dim );

    dec( rsi ); //i -= 1
    jne("BLOOPKITER", T_NEAR );

    L( "BCONSIDKLEFT" );
    // load k_left
    mov( rsi, ptr[ rsp + stack_off_k_iter_after_prefetch ] );
    test( rsi, rsi );
    je( "KFRINGE", T_NEAR );

    L( "KPARTIALLOOP" );
    //Partial k-left loop
    kernel_unroll( m_dim, n_dim );
    dec(rsi); //rsi -= 1
    jne( "KPARTIALLOOP", T_NEAR );

    L( "KFRINGE" );
    // load k_left
    mov( rsi, ptr[ rsp + stack_off_k_left ] );
    test( rsi, rsi );
    je( "BPOSTACCUM", T_NEAR );
    // k_fringe
    k_fringe_loop( m_dim, n_dim );
#else
    // load k_iter
    mov( rsi, ptr[ rsp + stack_off_k_iter_before_prefetch ] );
    test( rsi, rsi );
    je( "BPREFETCH", T_NEAR );
    L( "BLOOPKITER" );

    // Main k-unroll loop
    kernel_unroll( m_dim, n_dim );

    dec( rsi ); // i -= 1
    jne("BLOOPKITER", T_NEAR );

    L( "BPREFETCH" );

    prefetchC( m_dim, n_dim );

    mov( rsi, ptr[ rsp + stack_off_k_iter_after_prefetch ] );
    test( rsi, rsi );
    je( "BCONSIDKLEFT", T_NEAR );

    L( "AFTERPREFETCH" );

    kernel_unroll( m_dim, n_dim );

    dec( rsi );
    jne( "AFTERPREFETCH", T_NEAR );

    L( "BCONSIDKLEFT" );
    // load k_left
    mov( rsi, ptr[ rsp + stack_off_k_left ] );
    test( rsi, rsi );
    je( "BPOSTACCUM", T_NEAR );

    // k_fringe
    k_fringe_loop( m_dim, n_dim );
#endif

    L( "BPOSTACCUM" );

    // Generate alpha scaling code only when required.
    if( params->alpha_scale )
    {
        mov( rax, ptr[ rsp + stack_off_alpha ] );      // load address of alpha
        vbroadcastss( Zmm( alpha_reg ), ptr[ rax ] );

        scale_alpha( m_dim, n_dim );

    }

    mov( rbx, ptr[ rsp + stack_off_beta ] );
    vbroadcastss( Xmm( beta_reg ), ptr[ rbx ] );      // load address of beta

    // Zero out a register
    vxorps( Xmm( alpha_reg ), Xmm( alpha_reg ) );
    // cmp beta value with zero
    vucomiss( Xmm( beta_reg ), Xmm( alpha_reg ) );
    // if beta=0, skip beta scaling
    je( "BPOSTBETAOP", T_NEAR );

    // check if buf_downscale is NULL
    mov( rax, ptr[ rsp + stack_off_buf_downscale ] );
    cmp( rax, 0 );
    je( "BETAOP", T_NEAR );

    // Check if is_first_k is 0
    mov( rcx, ptr[ rsp + stack_off_postop +
                  offsetof( lpgemm_post_op_attr, is_first_k ) ] );
    test( rcx, rcx );
    je( "BETAOP", T_NEAR );

    L( "DOWNSCALEBETAOP" );
    vbroadcastss( Zmm( beta_reg ), ptr[ rbx ] );
    bf16_f32_beta_op( m_dim, n_dim );
    jmp( "BPOSTBETAOP", T_NEAR );

    L( "BETAOP" );
    mov( rcx, r12 );
    vbroadcastss( Zmm( beta_reg ), ptr[ rbx ] );
    f32_f32_beta_op( m_dim, n_dim );

    L( "BPOSTBETAOP" );

    // Check if is_last_k is 0
    mov( rcx, ptr[ rsp + stack_off_postop +
                  offsetof( lpgemm_post_op_attr, is_last_k ) ] );
    test(rcx, rcx);
    je( "POST_OPS_6x64_DISABLE", T_NEAR );

    post_op_label_lastk_safe_jump();


    L( "POST_OPS_BIAS_6x64" );

    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args2 ) ] );
    mov( bl, ptr[ rax ] );

    //check if op_args2 == 'R'
    cmp( bl, 0x52 );
    je("BIAS_ROW_MAJOR", T_NEAR );
    // check if op_args2 == 'r
    cmp( bl, 0x72 );
    je( "BIAS_ROW_MAJOR", T_NEAR );

    bias_col_major( m_dim, n_dim );
    jmp( "POST_BIAS", T_NEAR );

    L( "BIAS_ROW_MAJOR" );
    bias_row_major( m_dim, n_dim );

    L( "POST_BIAS" );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_RELU_6x64" );
    relu( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_RELU_SCALE_6x64" );
    relu_scale( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_GELU_TANH_6x64" );
    gelu_tanh( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_GELU_ERF_6x64" );
    gelu_erf( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_CLIP_6x64" );
    clip_f32( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_DOWNSCALE_6x64" );
    mov( rax, ptr[ rdx + offsetof( lpgemm_post_op, op_args2 ) ] );
    mov( bl, ptr[ rax ] );

    //check if op_args2 == 'R'
    cmp( bl, 0x52 );
    je("DOWNSCALE_ROW_MAJOR", T_NEAR );
    // check if op_args2 == 'r
    cmp( bl, 0x72 );
    je( "DOWNSCALE_ROW_MAJOR", T_NEAR );

    downscale_col_major( m_dim, n_dim );
    jmp( "POST_DOWNSCALE", T_NEAR );

    L( "DOWNSCALE_ROW_MAJOR" );
    downscale_row_major( m_dim, n_dim );

    L( "POST_DOWNSCALE" );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_MATRIX_ADD_6x64" );

    mov( rcx, ptr[ rsp + stack_off_postop +
                  offsetof( lpgemm_post_op_attr, c_stor_type ) ] );
    cmp( rcx, 4 );
    je( "BF16_MATADD", T_NEAR );
    f32_f32_matrix_add( m_dim, n_dim );
    jmp( "POST_MATADD", T_NEAR );
    L( "BF16_MATADD" );
    bf16_f32_matrix_add( m_dim, n_dim );
    L( "POST_MATADD" );

    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_SWISH_6x64" );
    swish( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_TANH_6x64" );
    tanh(m_dim, n_dim);
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_SIGMOID_6x64" );
    sigmoid( m_dim, n_dim );
    post_op_label_lastk_safe_jump_with_next_ptr();

    L( "POST_OPS_6x64_DISABLE" );

    // check if buf_downscale is NULL
    mov( rax, ptr[ rsp + stack_off_buf_downscale ] );
    cmp( rax, 0 );
    je( "F32_STORE", T_NEAR );

    // Check if is_last_k is 0
    mov( rcx, ptr[ rsp + stack_off_postop +
                  offsetof( lpgemm_post_op_attr, is_last_k ) ] );
    test( rcx, rcx );
    je( "F32_STORE", T_NEAR );

    L( "BF16_STORE" );
    //mov( rcx, ptr[rsp + stack_off_buf_downscale]);
    cvt_store_f32_bf16_mask( m_dim, n_dim );
    jmp( "END", T_NEAR );

    L( "F32_STORE" );
    mov( rcx, r12 );
    store_f32( m_dim, n_dim );

    L( "END" );

    if( params->m_loop )
    {
        mov(rax, ptr[ rsp + stack_off_ps_a ] );

        lea( r12, ptr[ r12 + rdi * 4 ] );
        lea( r12, ptr[ r12 + rdi * 2 ] );    // c_ii = r12 += 6*rs_c;

        lea(r14, ptr[ r14 + rax ] );         // a_ii = r14 += ps_a2

        //add(, m_dim );
        mov( rax, ptr[ rsp + stack_off_postop +
                       offsetof( lpgemm_post_op_attr, post_op_c_i ) ] );
        add( rax, m_dim);

        mov( ptr[ rsp + stack_off_postop +
                  offsetof( lpgemm_post_op_attr, post_op_c_i ) ], rax );

        mov( rdx, ptr[ rsp + stack_off_temp_list ] );

        dec(r11);
        jne("BLOOP6X64I", T_NEAR);
    }

    // release the space that is requested from stack
    add( rsp, 512 );

    // restore the callee-save registers.
    postamble();

    ret();

    align(64);
    L(tables);

    db(reinterpret_cast<uint8_t*>( &gelu_consts ), sizeof( gelu_consts ) );
    db(reinterpret_cast<uint8_t*>( &gelu_macros ), sizeof( gelu_macros ) );
    db(reinterpret_cast<uint8_t*>( &lpgemm_exp ), sizeof( lpgemm_exp ) );
    db(reinterpret_cast<uint8_t*>( &erf_consts ), sizeof( erf_consts ) );
    db(reinterpret_cast<uint8_t*>( &lpgemm_erf ), sizeof( lpgemm_erf ) );

}

const void (* bli_lpgemm_jit:: get_function ()const)( lpgemm_jit_params_t*,
                                                      lpgemm_post_op_attr*,
                                                      lpgemm_post_op* )
{
    return getCode<const void (*)( lpgemm_jit_params_t*,
                                   lpgemm_post_op_attr*,
                                   lpgemm_post_op*)>();
}

const void* bli_lpgemm_jit:: get_code ()const
{
    return getCode<const void (*)>();
}
dim_t bli_lpgemm_jit:: get_size ()
{
    return getSize();
}
