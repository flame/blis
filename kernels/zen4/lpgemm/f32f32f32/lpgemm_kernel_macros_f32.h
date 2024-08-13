/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_F32_SGEMM_KERN_MACROS_H
#define LPGEMM_F32_SGEMM_KERN_MACROS_H

#include "../gelu_avx512.h"
#include "../silu_avx512.h"
#include "../math_utils_avx512.h"

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32S_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_ps_mask( reg, zmm1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mul_ps( reg, relu_cmp_mask, reg, zmm2 ); \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32S_AVX512(reg, r, r2, x, z, dn, x_tanh, q) \
\
	GELU_TANH_F32_AVX512_DEF(reg, r, r2, x, z, dn, x_tanh, q); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32S_AVX512(reg, r, x, x_erf) \
\
	GELU_ERF_F32_AVX512_DEF(reg, r, x, x_erf); \

#define CLIP_F32S_AVX512(reg, min, max) \
\
	reg = _mm512_min_ps( _mm512_max_ps( reg, min ), max ); \

//Zero-out the given ZMM accumulator registers
#define ZERO_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3) \
      zmm0 = _mm512_setzero_ps(); \
      zmm1 = _mm512_setzero_ps(); \
      zmm2 = _mm512_setzero_ps(); \
      zmm3 = _mm512_setzero_ps();

// Zero-out the given ZMM accumulator registers
#define ZERO_ACC_XMM_4_REG(xmm0, xmm1, xmm2, xmm3) \
      xmm0 = _mm_setzero_ps(); \
      xmm1 = _mm_setzero_ps(); \
      xmm2 = _mm_setzero_ps(); \
      xmm3 = _mm_setzero_ps();

/*Multiply alpha with accumulator registers and store back*/
#define ALPHA_MUL_ACC_ZMM_4_REG(zmm0,zmm1,zmm2,zmm3,alpha) \
      zmm0 = _mm512_mul_ps(zmm0,alpha); \
      zmm1 = _mm512_mul_ps(zmm1,alpha); \
      zmm2 = _mm512_mul_ps(zmm2,alpha); \
      zmm3 = _mm512_mul_ps(zmm3,alpha);

// Matrix Add post-ops helper macros
#define F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \

#define F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_add_ps( scr2, zmm ## r_ind2 ); \

#define F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	zmm ## r_ind0 = _mm512_add_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_add_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_add_ps( scr2, zmm ## r_ind2 ); \
	zmm ## r_ind3 = _mm512_add_ps( scr3, zmm ## r_ind3 ); \

#define F32_F32_MATRIX_ADD_LOAD(mask,scr,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \

#define F32_F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_ADD_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define F32_F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_ADD_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define F32_F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define F32_F32_MATRIX_ADD_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_ADD_LOAD(k0,scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(k1,scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(k2,scr2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_ADD_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

// Matrix Mul post-ops helper macros
#define F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \

#define F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_mul_ps( scr2, zmm ## r_ind2 ); \

#define F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	zmm ## r_ind0 = _mm512_mul_ps( scr0, zmm ## r_ind0 ); \
	zmm ## r_ind1 = _mm512_mul_ps( scr1, zmm ## r_ind1 ); \
	zmm ## r_ind2 = _mm512_mul_ps( scr2, zmm ## r_ind2 ); \
	zmm ## r_ind3 = _mm512_mul_ps( scr3, zmm ## r_ind3 ); \

#define F32_F32_MATRIX_MUL_LOAD(mask,scr,m_ind,n_ind) \
	scr = _mm512_maskz_loadu_ps \
			( \
			  mask, \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 16 ) \
			); \

#define F32_F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_MATRIX_MUL_2COL(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define F32_F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2) \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_MATRIX_MUL_3COL(scr0,scr1,scr2,m_ind,r_ind0,r_ind1,r_ind2); \

#define F32_F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3) \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr1,m_ind,1); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr2,m_ind,2); \
	F32_F32_MATRIX_ADD_LOAD(_cvtu32_mask16( 0xFFFF ),scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#define F32_F32_MATRIX_MUL_4COL_MASK(k0,k1,k2,k3,r_ind0,r_ind1,r_ind2,r_ind3,scr0,scr1,scr2,scr3,m_ind) \
	F32_F32_MATRIX_MUL_LOAD(k0,scr0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD(k1,scr1,m_ind,1); \
	F32_F32_MATRIX_MUL_LOAD(k2,scr2,m_ind,2); \
	F32_F32_MATRIX_MUL_LOAD(k3,scr3,m_ind,3); \
	F32_MATRIX_MUL_4COL(scr0,scr1,scr2,scr3,m_ind,r_ind0,r_ind1,r_ind2,r_ind3); \

#endif //LPGEMM_F32_SGEMM_KERN_MACROS_H

