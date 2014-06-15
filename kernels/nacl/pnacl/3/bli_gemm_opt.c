/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#if PPAPI_RELEASE >= 36
typedef float v4sf __attribute__ ((vector_size(16)));

inline v4sf v4sf_splat(float x) {
	return (v4sf) { x, x, x, x };
}

inline v4sf v4sf_load(const float* a) {
	return *((const v4sf*)a);
}

inline void v4sf_store(float* a, v4sf x) {
	*((v4sf*)a) = x;
}

inline v4sf v4sf_zero() {
	return (v4sf) { 0.0f, 0.0f, 0.0f, 0.0f };
}
#endif

#if PPAPI_RELEASE >= 36 /* 8x4 blocks, SIMD enabled */
void bli_sgemm_opt_8x4(
	dim_t           k,
	float *restrict alpha,
	float *restrict a,
	float *restrict b,
	float *restrict beta,
	float *restrict c,
	inc_t           rs_c,
	inc_t           cs_c,
	auxinfo_t*      data)
{
	// Vectors for accummulating column 0, 1, 2, 3 (initialize to 0.0)
	v4sf abv0t = v4sf_zero(), abv1t = v4sf_zero(), abv2t = v4sf_zero(), abv3t = v4sf_zero();
	v4sf abv0b = v4sf_zero(), abv1b = v4sf_zero(), abv2b = v4sf_zero(), abv3b = v4sf_zero();
	for (dim_t i = 0; i < k; i += 1) {
		const v4sf avt = v4sf_load(a);
		const v4sf avb = v4sf_load(a+4);

		const v4sf bv_xxxx = v4sf_splat(b[0]);
		abv0t += avt * bv_xxxx;
		abv0b += avb * bv_xxxx;

		const v4sf bv_yyyy = v4sf_splat(b[1]);
		abv1t += avt * bv_yyyy;
		abv1b += avb * bv_yyyy;

		const v4sf bv_zzzz = v4sf_splat(b[2]);
		abv2t += avt * bv_zzzz;
		abv2b += avb * bv_zzzz;

		const v4sf bv_wwww = v4sf_splat(b[3]);
		abv3t += avt * bv_wwww;
		abv3b += avb * bv_wwww;

		a += 8;
		b += 4;
	}

	const v4sf alphav = v4sf_splat(*alpha);
	abv0t *= alphav;
	abv0b *= alphav;
	abv1t *= alphav;
	abv1b *= alphav;
	abv2t *= alphav;
	abv2b *= alphav;
	abv3t *= alphav;
	abv3b *= alphav;

	if (rs_c == 1) {
		v4sf cv0t = v4sf_load(&c[0*rs_c + 0*cs_c]);
		v4sf cv1t = v4sf_load(&c[0*rs_c + 1*cs_c]); 
		v4sf cv2t = v4sf_load(&c[0*rs_c + 2*cs_c]); 
		v4sf cv3t = v4sf_load(&c[0*rs_c + 3*cs_c]); 
		v4sf cv0b = v4sf_load(&c[4*rs_c + 0*cs_c]);
		v4sf cv1b = v4sf_load(&c[4*rs_c + 1*cs_c]); 
		v4sf cv2b = v4sf_load(&c[4*rs_c + 2*cs_c]); 
		v4sf cv3b = v4sf_load(&c[4*rs_c + 3*cs_c]); 

		const v4sf betav = v4sf_splat(*beta);
		cv0t = cv0t * betav + abv0t;
		cv1t = cv1t * betav + abv1t;
		cv2t = cv2t * betav + abv2t;
		cv3t = cv3t * betav + abv3t;
		cv0b = cv0b * betav + abv0b;
		cv1b = cv1b * betav + abv1b;
		cv2b = cv2b * betav + abv2b;
		cv3b = cv3b * betav + abv3b;

		v4sf_store(&c[0*rs_c + 0*cs_c], cv0t);
		v4sf_store(&c[0*rs_c + 1*cs_c], cv1t); 
		v4sf_store(&c[0*rs_c + 2*cs_c], cv2t); 
		v4sf_store(&c[0*rs_c + 3*cs_c], cv3t); 
		v4sf_store(&c[4*rs_c + 0*cs_c], cv0b);
		v4sf_store(&c[4*rs_c + 1*cs_c], cv1b); 
		v4sf_store(&c[4*rs_c + 2*cs_c], cv2b); 
		v4sf_store(&c[4*rs_c + 3*cs_c], cv3b); 
	} else {
		// Load columns 0, 1, 2, 3 (top part)
		v4sf cv0t = (v4sf){ c[0*rs_c + 0*cs_c], c[1*rs_c + 0*cs_c], c[2*rs_c + 0*cs_c], c[3*rs_c + 0*cs_c] };
		v4sf cv1t = (v4sf){ c[0*rs_c + 1*cs_c], c[1*rs_c + 1*cs_c], c[2*rs_c + 1*cs_c], c[3*rs_c + 1*cs_c] };
		v4sf cv2t = (v4sf){ c[0*rs_c + 2*cs_c], c[1*rs_c + 2*cs_c], c[2*rs_c + 2*cs_c], c[3*rs_c + 2*cs_c] };
		v4sf cv3t = (v4sf){ c[0*rs_c + 3*cs_c], c[1*rs_c + 3*cs_c], c[2*rs_c + 3*cs_c], c[3*rs_c + 3*cs_c] };
		// Load columns 0, 1, 2, 3 (bottom part)
		v4sf cv0b = (v4sf){ c[4*rs_c + 0*cs_c], c[5*rs_c + 0*cs_c], c[6*rs_c + 0*cs_c], c[7*rs_c + 0*cs_c] };
		v4sf cv1b = (v4sf){ c[4*rs_c + 1*cs_c], c[5*rs_c + 1*cs_c], c[6*rs_c + 1*cs_c], c[7*rs_c + 1*cs_c] };
		v4sf cv2b = (v4sf){ c[4*rs_c + 2*cs_c], c[5*rs_c + 2*cs_c], c[6*rs_c + 2*cs_c], c[7*rs_c + 2*cs_c] };
		v4sf cv3b = (v4sf){ c[4*rs_c + 3*cs_c], c[5*rs_c + 3*cs_c], c[6*rs_c + 3*cs_c], c[7*rs_c + 3*cs_c] };

		const v4sf betav = v4sf_splat(*beta);
		cv0t = cv0t * betav + abv0t;
		cv1t = cv1t * betav + abv1t;
		cv2t = cv2t * betav + abv2t;
		cv3t = cv3t * betav + abv3t;
		cv0b = cv0b * betav + abv0b;
		cv1b = cv1b * betav + abv1b;
		cv2b = cv2b * betav + abv2b;
		cv3b = cv3b * betav + abv3b;

		// Store column 0
		c[0*rs_c + 0*cs_c] = cv0t[0];
		c[1*rs_c + 0*cs_c] = cv0t[1];
		c[2*rs_c + 0*cs_c] = cv0t[2];
		c[3*rs_c + 0*cs_c] = cv0t[3];
		c[4*rs_c + 0*cs_c] = cv0b[0];
		c[5*rs_c + 0*cs_c] = cv0b[1];
		c[6*rs_c + 0*cs_c] = cv0b[2];
		c[7*rs_c + 0*cs_c] = cv0b[3];

		// Store column 1
		c[0*rs_c + 1*cs_c] = cv1t[0];
		c[1*rs_c + 1*cs_c] = cv1t[1];
		c[2*rs_c + 1*cs_c] = cv1t[2];
		c[3*rs_c + 1*cs_c] = cv1t[3];
		c[4*rs_c + 1*cs_c] = cv1b[0];
		c[5*rs_c + 1*cs_c] = cv1b[1];
		c[6*rs_c + 1*cs_c] = cv1b[2];
		c[7*rs_c + 1*cs_c] = cv1b[3];

		// Store column 2
		c[0*rs_c + 2*cs_c] = cv2t[0];
		c[1*rs_c + 2*cs_c] = cv2t[1];
		c[2*rs_c + 2*cs_c] = cv2t[2];
		c[3*rs_c + 2*cs_c] = cv2t[3];
		c[4*rs_c + 2*cs_c] = cv2b[0];
		c[5*rs_c + 2*cs_c] = cv2b[1];
		c[6*rs_c + 2*cs_c] = cv2b[2];
		c[7*rs_c + 2*cs_c] = cv2b[3];

		// Store column 3
		c[0*rs_c + 3*cs_c] = cv3t[0];
		c[1*rs_c + 3*cs_c] = cv3t[1];
		c[2*rs_c + 3*cs_c] = cv3t[2];
		c[3*rs_c + 3*cs_c] = cv3t[3];
		c[4*rs_c + 3*cs_c] = cv3b[0];
		c[5*rs_c + 3*cs_c] = cv3b[1];
		c[6*rs_c + 3*cs_c] = cv3b[2];
		c[7*rs_c + 3*cs_c] = cv3b[3];
	}
}
#else /* PPAPI_RELEASE < 36 case: 4x4 blocks, no SIMD */
void bli_sgemm_opt_4x4(
	dim_t           k,
	float *restrict alpha,
	float *restrict a,
	float *restrict b,
	float *restrict beta,
	float *restrict c,
	inc_t           rs_c,
	inc_t           cs_c,
	auxinfo_t*      data)
{
	/* Just call the reference implementation. */
	BLIS_SGEMM_UKERNEL_REF(
		k,
		alpha,
		a,
		b,
		beta,
		c,
		rs_c,
		cs_c,
		data);
}
#endif

void bli_dgemm_opt_4x4(
	dim_t            k,
	double *restrict alpha,
	double *restrict a,
	double *restrict b,
	double *restrict beta,
	double *restrict c,
	inc_t            rs_c,
	inc_t            cs_c,
	auxinfo_t*       data)
{
	/* Just call the reference implementation. */
	BLIS_DGEMM_UKERNEL_REF(
		k,
		alpha,
		a,
		b,
		beta,
		c,
		rs_c,
		cs_c,
		data);
}

void bli_cgemm_opt_4x4(
	dim_t              k,
	scomplex *restrict alpha,
	scomplex *restrict a,
	scomplex *restrict b,
	scomplex *restrict beta,
	scomplex *restrict c,
	inc_t              rs_c,
	inc_t              cs_c,
	auxinfo_t*         data)
{
	/* Just call the reference implementation. */
	BLIS_CGEMM_UKERNEL_REF(
		k,
		alpha,
		a,
		b,
		beta,
		c,
		rs_c,
		cs_c,
		data);
}

void bli_zgemm_opt_4x4(
	dim_t              k,
	dcomplex *restrict alpha,
	dcomplex *restrict a,
	dcomplex *restrict b,
	dcomplex *restrict beta,
	dcomplex *restrict c,
	inc_t              rs_c,
	inc_t              cs_c,
	auxinfo_t*         data)
{
	/* Just call the reference implementation. */
	BLIS_ZGEMM_UKERNEL_REF(
		k,
		alpha,
		a,
		b,
		beta,
		c,
		rs_c,
		cs_c,
		data);
}

