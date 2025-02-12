/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_F32_SGEMM_AVX2_KERN_MACROS_H
#define LPGEMM_F32_SGEMM_AVX2_KERN_MACROS_H

#include "../gelu_avx2.h"
#include "../silu_avx2.h"
#include "../sigmoid_avx2.h"
#include "../math_utils_avx2.h"

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32S_AVX2(reg, scale, zreg, scratch2) \
     scratch2 = _mm256_min_ps( reg, zreg ); /* <0 elems*/\
     reg = _mm256_max_ps( reg, zreg ); /* >=0 elems*/\
     scratch2 = _mm256_mul_ps( scratch2, scale ); /*scale <0 elems*/\
     reg = _mm256_or_ps( reg, scratch2 ); \

/* ReLU scale (Parametric ReLU):  f(x) = x, when x > 0 and f(x) = a*x when x <= 0 */
#define RELU_SCALE_OP_F32S_SSE(reg, scale, zreg, scratch2) \
     scratch2 = _mm_min_ps( reg, zreg ); /* <0 elems*/\
     reg = _mm_max_ps( reg, zreg ); /* >=0 elems*/\
     scratch2 = _mm_mul_ps( scratch2, scale ); /*scale <0 elems*/\
     reg = _mm_or_ps( reg, scratch2 ); \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32S_AVX2(reg, r, r2, x, z, dn, x_tanh, q) \
\
	GELU_TANH_F32_AVX2_DEF(reg, r, r2, x, z, dn, x_tanh, q); \

/* TANH GeLU (x) = 0.5* x * (1 + tanh ( 0.797884 * ( x + ( 0.044715 * x^3 ) ) ) )  */
#define GELU_TANH_F32S_SSE(reg, r, r2, x, z, dn, x_tanh, q) \
\
	GELU_TANH_F32_SSE_DEF(reg, r, r2, x, z, dn, x_tanh, q); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32S_AVX2(reg, r, x, x_erf) \
\
	GELU_ERF_F32_AVX2_DEF(reg, r, x, x_erf); \

/* ERF GeLU (x) = 0.5* x * (1 + erf (x * 0.707107 ))  */
#define GELU_ERF_F32S_SSE(reg, r, x, x_erf) \
\
	GELU_ERF_F32_SSE_DEF(reg, r, x, x_erf); \

#define CLIP_F32S_AVX2(reg, min, max) \
\
	reg = _mm256_min_ps( _mm256_max_ps( reg, min ), max ); \

#define CLIP_F32S_SSE(reg, min, max) \
\
	reg = _mm_min_ps( _mm_max_ps( reg, min ), max ); \

#define F32_SCL_MULRND_AVX2(reg, selector, zero_point)   \
\
	reg = _mm256_mul_ps(reg, selector);  \
	reg = _mm256_add_ps(reg, zero_point);  \

#define F32_SCL_MULRND_SSE(reg, selector, zero_point)   \
\
	reg = _mm_mul_ps(reg, selector);   \
	reg = _mm_add_ps(reg, zero_point);   \

//Zero-out the given YMM accumulator registers
#define ZERO_ACC_YMM_4_REG(ymm0,ymm1,ymm2,ymm3) \
      ymm0 = _mm256_setzero_ps(); \
      ymm1 = _mm256_setzero_ps(); \
      ymm2 = _mm256_setzero_ps(); \
      ymm3 = _mm256_setzero_ps();

//Zero-out the given XMM accumulator registers
#define ZERO_ACC_XMM_4_REG(xmm0,xmm1,xmm2,xmm3) \
      xmm0 = _mm_setzero_ps(); \
      xmm1 = _mm_setzero_ps(); \
      xmm2 = _mm_setzero_ps(); \
      xmm3 = _mm_setzero_ps();

/*Multiply alpha with accumulator registers and store back*/
#define ALPHA_MUL_ACC_YMM_4_REG(ymm0,ymm1,ymm2,ymm3,alpha) \
      ymm0 = _mm256_mul_ps(ymm0,alpha); \
      ymm1 = _mm256_mul_ps(ymm1,alpha); \
      ymm2 = _mm256_mul_ps(ymm2,alpha); \
      ymm3 = _mm256_mul_ps(ymm3,alpha);

/*Multiply alpha with accumulator registers and store back*/
#define ALPHA_MUL_ACC_XMM_4_REG(xmm0,xmm1,xmm2,xmm3,alpha) \
      xmm0 = _mm_mul_ps(xmm0,alpha); \
      xmm1 = _mm_mul_ps(xmm1,alpha); \
      xmm2 = _mm_mul_ps(xmm2,alpha); \
      xmm3 = _mm_mul_ps(xmm3,alpha);

/*Load C, Multiply with beta and add with A*B and store*/
#define F32_C_BNZ_8(cbuf,rs_c,ymm0,beta,ymm2) \
      ymm0 = _mm256_loadu_ps(cbuf); \
      ymm2 = _mm256_fmadd_ps(ymm0, beta, ymm2); \

/*Load C, Multiply with beta and add with A*B and store*/
#define F32_C_BNZ_4(cbuf,rs_c,xmm0,beta,xmm2) \
      xmm0 = _mm_loadu_ps(cbuf); \
      xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

/*Load C, Multiply with beta and add with A*B and store*/
#define F32_C_BNZ_2(cbuf,rs_c,xmm0,beta,xmm2) \
      xmm0 = ( __m128 )_mm_load_sd((const double*)cbuf); \
      xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

/*Load C, Multiply with beta and add with A*B and store*/
#define F32_C_BNZ_1(cbuf,rs_c,xmm0,beta,xmm2) \
      xmm0 = _mm_load_ss(cbuf); \
      xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

/*Load C from buf_downscale and convert to F32,
multiply with Beta, and add to alpha*A*B*/
#define BF16_F32_C_BNZ_8(m_ind,n_ind,ymm0,beta,ymm2) \
	ymm0 = (__m256)_mm256_sllv_epi32  \
			(  \
				_mm256_cvtepi16_epi32  \
				( \
					_mm_load_si128  \
					(  \
						( __m128i const* )( \
						( bfloat16* )post_ops_attr.buf_downscale + \
						( post_ops_attr.rs_c_downscale * \
							( post_ops_attr.post_op_c_i + m_ind ) ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 8 ) ) \
					)  \
				), _mm256_set1_epi32( 16 )  \
			); \
	ymm2 = _mm256_fmadd_ps(ymm0, beta, ymm2); \

/*Load C from buf_downscale and convert to F32,
multiply with Beta, and add to alpha*A*B*/
#define BF16_F32_C_BNZ_4(m_ind,n_ind,xmm0,beta,xmm2) \
	xmm0 =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					_mm_loadu_si128( (__m128i const*)( \
						( bfloat16* )post_ops_attr.buf_downscale + \
						( post_ops_attr.rs_c_downscale * \
							( post_ops_attr.post_op_c_i + m_ind ) ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 8 ) ))  \
				), 	_mm_set1_epi32(16) \
			); \
	xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

/*Load C from buf_downscale and convert to F32,
multiply with Beta, and add to alpha*A*B and strore*/
#define BF16_F32_C_BNZ_2(m_ind,n_ind,xmm0,beta,xmm2) \
	xmm0 =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					( __m128i )_mm_load_sd( (double const*)( \
						( bfloat16* )post_ops_attr.buf_downscale + \
						( post_ops_attr.rs_c_downscale * \
						( post_ops_attr.post_op_c_i + m_ind ) ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 8 ) ))  \
				), 	_mm_set1_epi32(16) \
			); \
	xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

/*Load C from buf_downscale and convert to F32,
multiply with Beta, and add to alpha*A*B*/
#define BF16_F32_C_BNZ_1(m_ind,n_ind,xmm0,beta,xmm2) \
	xmm0 =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					( __m128i )_mm_load_ss( (float const*)( \
							( bfloat16* )post_ops_attr.buf_downscale + \
							( post_ops_attr.rs_c_downscale * \
							( post_ops_attr.post_op_c_i + m_ind ) ) + \
							post_ops_attr.post_op_c_j + ( n_ind * 8 ) ))  \
				), 	_mm_set1_epi32(16) \
			); \
	xmm2 = _mm_fmadd_ps(xmm0, beta, xmm2); \

// Matrix Add post-ops helper macros
#define F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0) \
	xmm ## r_ind0 = _mm_add_ps( scr0, xmm ## r_ind0 ); \

#define F32_MATRIX_ADD_1COL_YMM(scr0,m_ind,r_ind0) \
	ymm ## r_ind0 = _mm256_add_ps( scr0, ymm ## r_ind0 ); \

#define F32_MATRIX_ADD_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1) \
	ymm ## r_ind0 = _mm256_add_ps( scr0, ymm ## r_ind0 ); \
	ymm ## r_ind1 = _mm256_add_ps( scr1, ymm ## r_ind1 ); \

#define F32_F32_MATRIX_ADD_LOAD_XMM_1ELE(scr,scl_fct,m_ind,n_ind) \
	scr = ( __m128 )_mm_load_ss \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 2 ) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_ADD_1COL_XMM_1ELE(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD_XMM_1ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_LOAD_XMM_2ELE(scr,scl_fct,m_ind,n_ind) \
	scr = ( __m128 )_mm_load_sd \
			( \
			  (double*)(matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 2 )) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_ADD_1COL_XMM_2ELE(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD_XMM_2ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_LOAD_XMM(scr,scl_fct,m_ind,n_ind) \
	scr = _mm_loadu_ps \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 4 ) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_ADD_1COL_XMM(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD_XMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_LOAD_YMM(scr,scl_fct,m_ind,n_ind) \
	scr = _mm256_loadu_ps \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 8 ) \
			); \
	scr = _mm256_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_ADD_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_YMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_ADD_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_ADD_LOAD_YMM(scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1); \

//Matrix-Add helpers for BF16 input.
#define BF16_F32_MATRIX_ADD_LOAD_YMM(scr,scl_fct,m_ind,n_ind) \
	scr =	(__m256)( _mm256_sllv_epi32  \
				(  \
					_mm256_cvtepi16_epi32  \
					( \
						_mm_loadu_si128 \
						( ( __m128i const* )( matptr + \
						( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 8 ) )  \
						) \
					), _mm256_set1_epi32( 16 )  \
				)  \
			); \
	scr = _mm256_mul_ps( scr, scl_fct ); \

#define BF16_F32_MATRIX_ADD_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	BF16_F32_MATRIX_ADD_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_ADD_LOAD_YMM(scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_ADD_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1); \

#define BF16_F32_MATRIX_ADD_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_YMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_ADD_LOAD_XMM(scr,scl_fct,m_ind,n_ind) \
	scr =	(__m128)_mm_sllv_epi32  \
			(  \
				_mm_cvtepi16_epi32  \
				( \
					( __m128i )_mm_load_sd \
					( \
						(double const*)( matptr + \
						( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 4 ) ) \
					)  \
				) , _mm_set1_epi32( 16 )  \
			);  \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define BF16_F32_MATRIX_ADD_1COL_XMM(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD_XMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_ADD_LOAD_XMM_2ELE(scr,scl_fct,m_ind,n_ind) \
	scr =	(__m128) _mm_sllv_epi32  \
			(  \
				_mm_cvtepi16_epi32  \
				( \
					( __m128i )_mm_load_ss \
					( \
						(float const*)(matptr + \
						( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
						post_ops_attr.post_op_c_j + ( n_ind * 2 ) ) \
					) \
				), _mm_set1_epi32( 16 )  \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define BF16_F32_MATRIX_ADD_1COL_XMM_2ELE(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD_XMM_2ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_ADD_LOAD_XMM_1ELE(scr,scl_fct,m_ind,n_ind) \
	{   \
		int16_t data_feeder[8] = {0};   \
		bfloat16 *post_op_ptr = ( bfloat16* )( matptr + \
						( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
						post_ops_attr.post_op_c_j );   \
		\
		for( dim_t i = 0; i < 1; i++ ) data_feeder[i] = *(post_op_ptr + i );  \
		scr = 	(__m128) _mm_sllv_epi32  \
				(  \
					_mm_cvtepi16_epi32  \
					( \
						( __m128i )_mm_loadu_si128 \
						( \
							( __m128i const* )( data_feeder ) \
						) \
					), _mm_set1_epi32( 16 )  \
				); \
		scr = _mm_mul_ps( scr, scl_fct ); \
	} \

#define BF16_F32_MATRIX_ADD_1COL_XMM_1ELE(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_ADD_LOAD_XMM_1ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_ADD_1COL_XMM(scr0,m_ind,r_ind0); \

// Matrix Mul post-ops helper macros
#define F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0) \
	xmm ## r_ind0 = _mm_mul_ps( scr0, xmm ## r_ind0 ); \

#define F32_MATRIX_MUL_1COL_YMM(scr0,m_ind,r_ind0) \
	ymm ## r_ind0 = _mm256_mul_ps( scr0, ymm ## r_ind0 ); \

#define F32_MATRIX_MUL_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1) \
	ymm ## r_ind0 = _mm256_mul_ps( scr0, ymm ## r_ind0 ); \
	ymm ## r_ind1 = _mm256_mul_ps( scr1, ymm ## r_ind1 ); \

#define F32_F32_MATRIX_MUL_LOAD_XMM_1ELE(scr,scl_fct,m_ind,n_ind) \
	scr = ( __m128 )_mm_load_ss \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 2 ) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_MUL_1COL_XMM_1ELE(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD_XMM_1ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_LOAD_XMM_2ELE(scr,scl_fct,m_ind,n_ind) \
	scr = ( __m128 )_mm_load_sd \
			( \
			  (double*)(matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 2 )) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_MUL_1COL_XMM_2ELE(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD_XMM_2ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_LOAD_XMM(scr,scl_fct,m_ind,n_ind) \
	scr = _mm_loadu_ps \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 4 ) \
			); \
	scr = _mm_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_MUL_1COL_XMM(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD_XMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_LOAD_YMM(scr,scl_fct,m_ind,n_ind) \
	scr = _mm256_loadu_ps \
			( \
			  matptr + ( ( post_ops_attr.post_op_c_i + m_ind ) * ldm ) + \
			  post_ops_attr.post_op_c_j + ( n_ind * 8 ) \
			); \
	scr = _mm256_mul_ps( scr, scl_fct ); \

#define F32_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	F32_F32_MATRIX_MUL_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_YMM(scr0,m_ind,r_ind0); \

#define F32_F32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	F32_F32_MATRIX_MUL_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_F32_MATRIX_MUL_LOAD_YMM(scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1); \

//BF16->F32 Matrix Mul Helpers
#define BF16_F32_MATRIX_MUL_LOAD_XMM_1ELE(scr,scl_fct,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD_XMM_1ELE(scr,scl_fct,m_ind,n_ind) \

#define BF16_F32_MATRIX_MUL_1COL_XMM_1ELE(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD_XMM_1ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_LOAD_XMM_2ELE(scr,scl_fct,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD_XMM_2ELE(scr,scl_fct,m_ind,n_ind) \

#define BF16_F32_MATRIX_MUL_1COL_XMM_2ELE(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD_XMM_2ELE(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_LOAD_XMM(scr,scl_fct,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD_XMM(scr,scl_fct,m_ind,n_ind) \

#define BF16_F32_MATRIX_MUL_1COL_XMM(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD_XMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_XMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_LOAD_YMM(scr0,scl_fct0,m_ind,n_ind) \
	BF16_F32_MATRIX_ADD_LOAD_YMM(scr0,scl_fct0,m_ind,n_ind); \

#define BF16_F32_MATRIX_MUL_1COL(scr0,scl_fct0,m_ind,r_ind0) \
	BF16_F32_MATRIX_MUL_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	F32_MATRIX_MUL_1COL_YMM(scr0,m_ind,r_ind0); \

#define BF16_F32_MATRIX_MUL_2COL(scr0,scr1,scl_fct0,scl_fct1,m_ind,r_ind0,r_ind1) \
	BF16_F32_MATRIX_MUL_LOAD_YMM(scr0,scl_fct0,m_ind,0); \
	BF16_F32_MATRIX_MUL_LOAD_YMM(scr1,scl_fct1,m_ind,1); \
	F32_MATRIX_MUL_2COL_YMM(scr0,scr1,m_ind,r_ind0,r_ind1); \

// TANH
#define TANH_F32S_AVX2(reg, r, r2, x, z, dn, q) \
\
	TANHF_AVX2(reg, r, r2, x, z, dn, q); \

// TANH
#define TANH_F32S_SSE(reg, r, r2, x, z, dn, q) \
\
	TANHF_SSE(reg, r, r2, x, z, dn, q);

//BF16 -> F32 helper
#define CVT_BF16_F32_SHIFT_AVX2(in) \
	(__m256)((__m256i)_mm256_sllv_epi32( _mm256_cvtepi16_epi32 (in),\
			_mm256_set1_epi32( 16 ) ) );

//BF16->F32 BIAS helpers
#define BF16_F32_BIAS_LOAD_AVX2(scr,n_ind) \
	scr = (__m256)( _mm256_sllv_epi32  \
					(  \
						_mm256_cvtepi16_epi32  \
						( \
							_mm_load_si128  \
							(  \
								( __m128i const* )( \
								( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
								post_ops_attr.post_op_c_j + ( n_ind * 8 ) ) \
							)  \
						), _mm256_set1_epi32( 16 )  \
					) \
				); \

#define BF16_F32_BIAS_BCAST_AVX2(scr,m_ind)  \
	scr = (__m256)( _mm256_sllv_epi32  \
				(  \
					_mm256_cvtepi16_epi32 \
					( \
						_mm_set1_epi16 \
						(  \
							*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) +  \
							post_ops_attr.post_op_c_i + m_ind )   \
						) \
					), _mm256_set1_epi32( 16 )  \
				) \
			);

#define BF16_F32_BIAS_LOAD_4BF16_AVX2(scr, idx) \
{  \
	scr =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					(__m128i)_mm_load_sd( (double const*) \
						(  ( bfloat16* )post_ops_list_temp->op_args1 + \
						post_ops_attr.post_op_c_j + ( idx * 4) ) )  \
				), 	_mm_set1_epi32(16) \
			); \
}

#define BF16_F32_BIAS_LOAD_2BF16_AVX2(scr, idx) \
{  \
	scr =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					(__m128i)_mm_load_ss( (float const*) \
						( ( bfloat16* )post_ops_list_temp->op_args1 + \
						post_ops_attr.post_op_c_j + ( idx * 2) )  )\
				), 	_mm_set1_epi32(16) \
			); \
}

#define BF16_F32_BIAS_LOAD_1BF16_AVX2(scr, idx) \
{  \
	bfloat16 data_feeder[8] = {0};  \
	memcpy( data_feeder, (bfloat16* )post_ops_list_temp->op_args1 + \
			post_ops_attr.post_op_c_j + ( idx * 1 ) , \
			sizeof( bfloat16 ) );  \
	scr =	(__m128)_mm_sllv_epi32  \
			( \
				_mm_cvtepi16_epi32  \
				( \
					(__m128i)_mm_loadu_si128( (__m128i const*)data_feeder )  \
				), 	_mm_set1_epi32(16) \
			); \
}

#define BF16_F32_BIAS_BCAST_LT4BF16_AVX2(scr,m_ind)  \
{ \
    scr = (__m128)_mm_sllv_epi32  \
            (  \
                _mm_cvtepi16_epi32 \
                ( \
                    _mm_set1_epi16 \
                    (  \
                        *( ( ( bfloat16* )post_ops_list_temp->op_args1 ) + \
                            post_ops_attr.post_op_c_i + m_ind )   \
                    ) \
                ), _mm_set1_epi32( 16 )  \
            );   \
}


#define STORE_F32_BF16_YMM( reg, m_ind, n_ind ) \
{ \
	_mm256_storeu_ps((float*)temp, reg); \
	dest = ( bfloat16* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j + ( n_ind * 8 ); \
	for(i = 0; i < 8; i++) \
	{ \
		tlsb = ( temp[i] & ( uint32_t )0x00010000 ) > 16; \
		rounded = temp[i] + ( uint32_t )0x00007FFF + tlsb; \
		memcpy( (dest+i), ((char *)(&rounded))+2, sizeof(bfloat16)); \
	} \
}

#define STORE_F32_BF16_4XMM( reg, m_ind, n_ind ) \
{ \
	_mm_storeu_ps((float*)temp, reg); \
	dest = ( bfloat16* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j + ( n_ind * 4 ); \
	for(i = 0; i < 4; i++) \
	{ \
		tlsb = ( temp[i] & ( uint32_t )0x00010000 ) > 16; \
		rounded = temp[i] + ( uint32_t )0x00007FFF + tlsb; \
		memcpy( (dest+i), ((char *)(&rounded))+2, sizeof(bfloat16)); \
	} \
}

#define STORE_F32_BF16_2XMM( reg, m_ind, n_ind ) \
{ \
	_mm_store_sd((double*)temp,  ( __m128d )reg); \
	dest = ( bfloat16* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j + ( n_ind * 2 ); \
	for(i = 0; i < 2; i++) \
	{ \
		tlsb = ( temp[i] & ( uint32_t )0x00010000 ) > 16; \
		rounded = temp[i] + ( uint32_t )0x00007FFF + tlsb; \
		memcpy( (dest+i), ((char *)(&rounded))+2, sizeof(bfloat16)); \
	} \
}

#define STORE_F32_BF16_1XMM( reg, m_ind, n_ind ) \
{ \
	_mm_store_ss((float*)temp, reg); \
	dest = ( bfloat16* )post_ops_attr.buf_downscale + \
		( post_ops_attr.rs_c_downscale * ( post_ops_attr.post_op_c_i + m_ind ) ) + \
		post_ops_attr.post_op_c_j + ( n_ind * 8 ); \
	for(i = 0; i < 1; i++) \
	{ \
		tlsb = ( temp[i] & ( uint32_t )0x00010000 ) > 16; \
		rounded = temp[i] + ( uint32_t )0x00007FFF + tlsb; \
		memcpy( (dest+i), ((char *)(&rounded))+2, sizeof(bfloat16)); \
	} \
}

/*Downscale Zeropoint BF16->F32 Helpers*/
#define BF16_F32_ZP_SCALAR_BCAST_AVX2(scr)  \
	scr = (__m256)( _mm256_sllv_epi32  \
				(  \
					_mm256_cvtepi16_epi32 \
					( \
						_mm_set1_epi16 \
						(  \
							*( ( bfloat16* )post_ops_list_temp->op_args1 )  \
						) \
					), _mm256_set1_epi32( 16 )  \
				) \
			);

#define BF16_F32_ZP_VECTOR_BCAST_AVX2(scr, m_ind)  \
	BF16_F32_BIAS_BCAST_AVX2(scr,m_ind);

#define BF16_F32_ZP_VECTOR_LOAD_AVX2(scr,n_ind)  \
	BF16_F32_BIAS_LOAD_AVX2(scr,n_ind)

#define BF16_F32_ZP_SCALAR_BCAST_SSE(scr)  \
	scr = (__m128)_mm_sllv_epi32  \
			(  \
				_mm_cvtepi16_epi32 \
				( \
					_mm_set1_epi16 \
					(  \
						*( ( ( bfloat16* )post_ops_list_temp->op_args1 ) )   \
					) \
				), _mm_set1_epi32( 16 )  \
			);   \

#define BF16_F32_ZP_VECTOR_BCAST_SSE(scr, m_ind)  \
	BF16_F32_BIAS_BCAST_LT4BF16_AVX2(scr,m_ind);

#define BF16_F32_ZP_VECTOR_4LOAD_SSE(scr,idx) \
	BF16_F32_BIAS_LOAD_4BF16_AVX2(scr,idx); \

#define BF16_F32_ZP_VECTOR_2LOAD_SSE(scr,idx) \
	BF16_F32_BIAS_LOAD_2BF16_AVX2(scr, idx) \

#define BF16_F32_ZP_VECTOR_1LOAD_SSE(scr,idx) \
	BF16_F32_BIAS_LOAD_1BF16_AVX2(scr, idx) \

// BF16->F32 Store mask helper
#define GET_STORE_MASK(mask,store_mask)   \
{  \
	int32_t mask_vec[8] = {0}; \
	for( dim_t i = 0; i < mask; i++ ) mask_vec[i] = -1;  \
	store_mask = _mm256_loadu_si256((__m256i const *)mask_vec); \
}

#endif //LPGEMM_F32_SGEMM_AVX2_KERN_MACROS_H
