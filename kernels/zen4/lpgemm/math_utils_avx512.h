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
#ifndef AOCL_LPGEMM_MATH_UTILS_AVX512_H
#define AOCL_LPGEMM_MATH_UTILS_AVX512_H

//constants for exp function
#define lpgemm_exp_c0 0x1.0000014439a91p0
#define lpgemm_exp_c1 0x1.62e43170e3344p-1
#define lpgemm_exp_c2 0x1.ebf906bc4c115p-3
#define lpgemm_exp_c3 0x1.c6ae2bb88c0c8p-5
#define lpgemm_exp_c4 0x1.3d1079db4ef69p-7
#define lpgemm_exp_c5 0x1.5f8905cb0cc4ep-10

#define TBL_LN2 0x1.71547652b82fep+0
#define EXPF_HUGE 0x1.8p+23
#define EXPF_MIN -88.0f
#define EXPF_MAX 88.0f
#define inf 1.0/0.0
#define sign_bit_mask -2147483648

//Trignometric EXP, TANH and ERF functions for AVX512

#define POLY_EVAL_6_AVX512(r, r2, z) \
    r2 = _mm512_mul_ps (r, r); \
    z = _mm512_fmadd_ps (r2, _mm512_fmadd_ps (r, _mm512_set1_ps(lpgemm_exp_c3), _mm512_set1_ps(lpgemm_exp_c2)), \
        _mm512_fmadd_ps (r, _mm512_set1_ps(lpgemm_exp_c1), _mm512_set1_ps(lpgemm_exp_c0))); \
    r2 = _mm512_mul_ps (r2, r2); \
    r = _mm512_fmadd_ps (r2, _mm512_fmadd_ps (r, _mm512_set1_ps(lpgemm_exp_c5), _mm512_set1_ps(lpgemm_exp_c4)), z); \

// Require in and out registers to be different. x : in, q : out.
#define EXPF_AVX512(x, r, r2, z, dn, q) \
    z = _mm512_mul_ps (x, _mm512_set1_ps(TBL_LN2));	\
	dn = _mm512_add_ps (z , _mm512_set1_ps(EXPF_HUGE));  \
    r = _mm512_sub_ps (z , _mm512_sub_ps (dn , _mm512_set1_ps(EXPF_HUGE)));  \
\
    POLY_EVAL_6_AVX512 (r, r2, z); \
\
    q = _mm512_add_epi32((__m512i) (r), _mm512_sllv_epi32 ((__m512i)dn, _mm512_set1_epi32 (23)) ); \
    q = _mm512_mask_and_epi32 ((__m512i) q, _mm512_cmpnle_ps_mask ( _mm512_set1_ps(EXPF_MIN), x), (__m512i)q, _mm512_set1_epi32(0)); \
    q = _mm512_mask_xor_epi32 ((__m512i)_mm512_set1_ps(inf), _mm512_cmpnle_ps_mask ( _mm512_set1_ps(EXPF_MAX), x), (__m512i)q, _mm512_set1_epi32(0));

#define TANHF_AVX512(x_tanh, r, r2, x, z, dn, q) \
    x = _mm512_mul_ps (_mm512_abs_ps (x_tanh), _mm512_set1_ps(-2) ); \
\
    EXPF_AVX512(x, r, r2, z, dn, q); \
\
    z =  _mm512_add_ps ((__m512)q, _mm512_set1_ps(-1)); \
    z = _mm512_div_ps (z, _mm512_add_ps (z, _mm512_set1_ps(2))); \
    z = _mm512_mul_ps (z, _mm512_set1_ps(-1)); \
    x_tanh = (__m512)(_mm512_xor_epi32 (_mm512_and_epi32 ((__m512i)x_tanh, (_mm512_set1_epi32(sign_bit_mask))), (__m512i)z)) ;

/*
    The erf function implementation is taken from the AMD AOCL LIBM library.
    https://github.com/AMD-AOCL/aocl-libm/blob/amd-main/src/optimized/erff.c
*/

#define as_v16_f32_u32(x) _mm512_castsi512_ps(x)
#define as_v16_u32_f32(x) _mm512_castps_si512(x)

#define erf512_c0  _mm512_set1_pd((0x1.20dd7890d27e1cec99fce48c29cp0))
#define erf512_c1  _mm512_set1_pd((-0x1.ab4bed70f238422edeeba9c558p-16))
#define erf512_c2  _mm512_set1_pd((-0x1.80a1bd5878e0b0689c5ff4fcdd4p-2))
#define erf512_c3  _mm512_set1_pd((-0x1.07cb4cde6a7d9528c8a732990e4p-8))
#define erf512_c4  _mm512_set1_pd((0x1.092cba598f96f00ddc5854cf7cp-3))
#define erf512_c5  _mm512_set1_pd((-0x1.51f0ce4ac87c55f11f685864714p-5))
#define erf512_c6  _mm512_set1_pd((0x1.4101f320bf8bc4d41c228faaa6cp-5))
#define erf512_c7  _mm512_set1_pd((-0x1.2300882a7d1b712726997de80ep-4))
#define erf512_c8  _mm512_set1_pd((0x1.d45745fff0e4b6d0604a9ab6284p-5))
#define erf512_c9  _mm512_set1_pd((-0x1.9eb1491956e31ded96176d7c8acp-6))
#define erf512_c10  _mm512_set1_pd((0x1.b9183fc75d326b9044bc63c9694p-8))
#define erf512_c11  _mm512_set1_pd((-0x1.10e8f8c89ad8645e7d769cd596cp-10))
#define erf512_c12  _mm512_set1_pd((0x1.224ffc80cc19957a48ecedad6c8p-14))
#define erf512_c13  _mm512_set1_pd((0x1.12a30f42c71308321e7e7cb0174p-18))
#define erf512_c14  _mm512_set1_pd((-0x1.155445e2e006723066d72d22ddcp-20))
#define erf512_c15  _mm512_set1_pd((0x1.c6a4181da4ef76f22bd39bb5dcp-25))

#define ERF512_UBOUND    (0x407AD447)  // 3.402823466E+38F
#define ERF512_BOUND     _mm512_set1_ps((float)(3.91920638084411621F))

typedef union {
    float    f;
    int  i;
    unsigned int u;
} flt32_t;

static inline unsigned int
asuint32(float f)
{
	flt32_t fl = {.f = f};
	return fl.u;
}

#define POLY_EVAL_HORNER_16_0_AVX512(x, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9,\
                                     c10, c11, c12, c13, c14, c15) ({            \
    __typeof(x) q = _mm512_mul_pd(x , _mm512_fmadd_pd(_mm512_fmadd_pd( \
                    _mm512_fmadd_pd(_mm512_fmadd_pd(_mm512_fmadd_pd( \
                    _mm512_fmadd_pd(_mm512_fmadd_pd(_mm512_fmadd_pd( \
                    _mm512_fmadd_pd(_mm512_fmadd_pd(_mm512_fmadd_pd( \
                    _mm512_fmadd_pd(_mm512_fmadd_pd(_mm512_fmadd_pd( \
                    _mm512_fmadd_pd( c15, x, c14), x, c13), x, c12), \
                    x, c11), x, c10), x, c9), x, c8), x, c7), x, c6), x, \
                    c5), x, c4), x, c3), x, c2), x, c1), x, c0)); \
q; \
})

//ERF_AOCL Macro
#define ERF_AOCL_AVX512(y, r) \
{ \
    __m512 absr = _mm512_abs_ps(r); \
    uint32_t uxmax = asuint32(_mm512_reduce_max_ps(absr)); \
    __m512d _y1d = _mm512_cvtps_pd(_mm512_extractf32x8_ps(absr, 0)); \
    __m512d _y2d = _mm512_cvtps_pd(_mm512_extractf32x8_ps(absr, 1)); \
 \
    _y1d = POLY_EVAL_HORNER_16_0_AVX512(_y1d, erf512_c0, erf512_c1, erf512_c2,\
                                        erf512_c3, erf512_c4, erf512_c5, erf512_c6, \
                                        erf512_c7, erf512_c8, erf512_c9, erf512_c10,\
                                        erf512_c11, erf512_c12, erf512_c13, \
                                        erf512_c14, erf512_c15); \
    _y2d = POLY_EVAL_HORNER_16_0_AVX512(_y2d, erf512_c0, erf512_c1, erf512_c2,\
                                        erf512_c3, erf512_c4, erf512_c5, erf512_c6, \
                                        erf512_c7, erf512_c8, erf512_c9, erf512_c10,\
                                        erf512_c11, erf512_c12, erf512_c13, \
                                        erf512_c14, erf512_c15); \
\
    y = _mm512_insertf32x8(y, _mm512_cvtpd_ps(_y1d), 0); \
    y = _mm512_insertf32x8(y, _mm512_cvtpd_ps(_y2d), 1); \
\
    __m512i sign = _mm512_and_epi32(_mm512_castps_si512(r), \
                   _mm512_set1_epi32((unsigned int)0x80000000)); \
\
    y = as_v16_f32_u32(sign | as_v16_u32_f32(y)); \
    if (uxmax > ERF512_UBOUND) { \
        __mmask16 mask = _mm512_cmp_ps_mask((ERF512_BOUND), absr, _CMP_LT_OQ); \
        __m512 fONE = _mm512_set1_ps(1.0f); \
        y = _mm512_mask_blend_ps(mask, y, fONE); \
        y = as_v16_f32_u32(sign | as_v16_u32_f32(y)); \
    } \
} \


#endif // AOCL_LPGEMM_MATH_UTILS_AVX512_H
