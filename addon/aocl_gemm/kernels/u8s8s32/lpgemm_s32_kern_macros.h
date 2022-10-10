/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_S32_KERN_MACROS_H
#define LPGEMM_S32_KERN_MACROS_H
#define S8_MIN  (-128)
#define S8_MAX  (+127)

#define RELU_SCALE_OP_S32_AVX512(reg) \
	/* Generate indenx of elements <= 0.*/ \
	relu_cmp_mask = _mm512_cmple_epi32_mask( reg, selector1 ); \
 \
	/* Apply scaling on for <= 0 elements.*/ \
	reg = _mm512_mask_mullo_epi32( reg, relu_cmp_mask, reg, selector2 ); \

#define CVT_MULRND_CVT32_CVT8(reg,selector,m_ind,n_ind) \
	_mm_storeu_epi8 \
	( \
	  ( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + m_ind ) ) + post_op_c_j + ( n_ind * 16 ), \
	  _mm512_cvtepi32_epi8 \
	  ( \
		_mm512_cvtps_epi32 \
		( \
		  _mm512_min_ps \
		  ( \
			_mm512_max_ps \
			( \
			  _mm512_mul_round_ps \
			  ( \
				_mm512_cvtepi32_ps( reg ), \
				( __m512 )selector, \
				( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
			  ) \
			  , _mm512_set1_ps (( float )S8_MIN) \
			) \
			, _mm512_set1_ps (( float )S8_MAX) \
		  ) \
		) \
	  ) \
	) \

#define CVT_MULRND_CVT32_CVT8_LT16(reg,selector,m_ind,n_ind) \
	_mm_storeu_epi8 \
	( \
	  buf0, \
	  _mm512_cvtepi32_epi8 \
	  ( \
		_mm512_cvtps_epi32 \
		( \
		  _mm512_min_ps \
		  ( \
			_mm512_max_ps \
			( \
			  _mm512_mul_round_ps \
			  ( \
				_mm512_cvtepi32_ps( reg ), \
				( __m512 )selector, \
				( _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC ) \
			  ) \
			  , _mm512_set1_ps (( float )S8_MIN) \
			) \
			, _mm512_set1_ps (( float )S8_MAX) \
		  ) \
		) \
	  ) \
	); \
	memcpy( ( int8_t* )post_ops_list_temp->op_args3 + \
	  ( rs_c_downscale * ( post_op_c_i + m_ind ) ) + post_op_c_j + \
	  ( n_ind * 16 ) , buf0, ( n0_rem * sizeof( int8_t ) ) ); \

#endif // LPGEMM_S32_KERN_MACROS_H
