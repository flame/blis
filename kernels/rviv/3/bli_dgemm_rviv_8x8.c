/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin
   Copyright (C) 2026, Hugo Meiland

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
// Fixed 8x8 dgemm microkernel using RVV LMUL=2 accumulators.
// Selected for VLEN >= 256 to keep MR/MC modest vs 4vx4.

#include "bli_rviv_utils.h"
#include <riscv_vector.h>

void bli_dgemm_rviv_8x8
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const dim_t mr = 8, nr = 8;
	GEMM_UKR_SETUP_CT( d, mr, nr, false );
	assert( rs_c == 1 );

	const double* restrict ap = a;
	const double* restrict bp = b;
	double*       restrict cp = c;
	const double alpha_r = *(const double*)alpha;
	const double beta_r  = *(const double*)beta;
	const size_t vl = __riscv_vsetvl_e64m2(8);
	// Requires VLEN >= 256 so that LMUL=2 can hold 8 doubles.
	assert( vl == 8 );

	vfloat64m2_t ab0 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab1 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab2 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab3 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab4 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab5 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab6 = __riscv_vfmv_v_f_f64m2(0.0, vl);
	vfloat64m2_t ab7 = __riscv_vfmv_v_f_f64m2(0.0, vl);

	dim_t p = k;
	for ( ; p >= 8; p -= 8 )
	{
		__builtin_prefetch(ap + 64, 0, 3);
		__builtin_prefetch(bp + 64, 0, 3);
		__builtin_prefetch(ap + 128, 0, 1);
		__builtin_prefetch(bp + 128, 0, 1);

		vfloat64m2_t a0 = __riscv_vle64_v_f64m2(ap +  0, vl);
		vfloat64m2_t a1 = __riscv_vle64_v_f64m2(ap +  8, vl);
		vfloat64m2_t a2 = __riscv_vle64_v_f64m2(ap + 16, vl);
		vfloat64m2_t a3 = __riscv_vle64_v_f64m2(ap + 24, vl);
		vfloat64m2_t a4 = __riscv_vle64_v_f64m2(ap + 32, vl);
		vfloat64m2_t a5 = __riscv_vle64_v_f64m2(ap + 40, vl);
		vfloat64m2_t a6 = __riscv_vle64_v_f64m2(ap + 48, vl);
		vfloat64m2_t a7 = __riscv_vle64_v_f64m2(ap + 56, vl);

		double b00=bp[0], b01=bp[1], b02=bp[2], b03=bp[3];
		double b04=bp[4], b05=bp[5], b06=bp[6], b07=bp[7];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b00, a0, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b01, a0, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b02, a0, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b03, a0, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b04, a0, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b05, a0, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b06, a0, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b07, a0, vl);

		double b10=bp[8], b11=bp[9], b12=bp[10], b13=bp[11];
		double b14=bp[12], b15=bp[13], b16=bp[14], b17=bp[15];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b10, a1, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b11, a1, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b12, a1, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b13, a1, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b14, a1, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b15, a1, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b16, a1, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b17, a1, vl);

		double b20=bp[16], b21=bp[17], b22=bp[18], b23=bp[19];
		double b24=bp[20], b25=bp[21], b26=bp[22], b27=bp[23];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b20, a2, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b21, a2, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b22, a2, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b23, a2, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b24, a2, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b25, a2, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b26, a2, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b27, a2, vl);

		double b30=bp[24], b31=bp[25], b32=bp[26], b33=bp[27];
		double b34=bp[28], b35=bp[29], b36=bp[30], b37=bp[31];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b30, a3, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b31, a3, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b32, a3, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b33, a3, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b34, a3, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b35, a3, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b36, a3, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b37, a3, vl);

		double b40=bp[32], b41=bp[33], b42=bp[34], b43=bp[35];
		double b44=bp[36], b45=bp[37], b46=bp[38], b47=bp[39];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b40, a4, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b41, a4, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b42, a4, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b43, a4, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b44, a4, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b45, a4, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b46, a4, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b47, a4, vl);

		double b50=bp[40], b51=bp[41], b52=bp[42], b53=bp[43];
		double b54=bp[44], b55=bp[45], b56=bp[46], b57=bp[47];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b50, a5, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b51, a5, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b52, a5, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b53, a5, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b54, a5, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b55, a5, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b56, a5, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b57, a5, vl);

		double b60=bp[48], b61=bp[49], b62=bp[50], b63=bp[51];
		double b64=bp[52], b65=bp[53], b66=bp[54], b67=bp[55];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b60, a6, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b61, a6, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b62, a6, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b63, a6, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b64, a6, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b65, a6, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b66, a6, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b67, a6, vl);

		double b70=bp[56], b71=bp[57], b72=bp[58], b73=bp[59];
		double b74=bp[60], b75=bp[61], b76=bp[62], b77=bp[63];
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, b70, a7, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, b71, a7, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, b72, a7, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, b73, a7, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, b74, a7, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, b75, a7, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, b76, a7, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, b77, a7, vl);

		ap += 64;
		bp += 64;
	}
	for ( ; p >= 4; p -= 4 )
	{
		vfloat64m2_t a0 = __riscv_vle64_v_f64m2(ap +  0, vl);
		vfloat64m2_t a1 = __riscv_vle64_v_f64m2(ap +  8, vl);
		vfloat64m2_t a2 = __riscv_vle64_v_f64m2(ap + 16, vl);
		vfloat64m2_t a3 = __riscv_vle64_v_f64m2(ap + 24, vl);
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, bp[0], a0, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, bp[1], a0, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, bp[2], a0, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, bp[3], a0, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, bp[4], a0, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, bp[5], a0, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, bp[6], a0, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, bp[7], a0, vl);
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, bp[8],  a1, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, bp[9],  a1, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, bp[10], a1, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, bp[11], a1, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, bp[12], a1, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, bp[13], a1, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, bp[14], a1, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, bp[15], a1, vl);
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, bp[16], a2, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, bp[17], a2, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, bp[18], a2, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, bp[19], a2, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, bp[20], a2, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, bp[21], a2, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, bp[22], a2, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, bp[23], a2, vl);
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, bp[24], a3, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, bp[25], a3, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, bp[26], a3, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, bp[27], a3, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, bp[28], a3, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, bp[29], a3, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, bp[30], a3, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, bp[31], a3, vl);
		ap += 32; bp += 32;
	}
	for ( ; p > 0; --p )
	{
		vfloat64m2_t a0 = __riscv_vle64_v_f64m2(ap, vl);
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, bp[0], a0, vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, bp[1], a0, vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, bp[2], a0, vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, bp[3], a0, vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, bp[4], a0, vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, bp[5], a0, vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, bp[6], a0, vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, bp[7], a0, vl);
		ap += 8; bp += 8;
	}

	if ( alpha_r != 1.0 )
	{
		ab0 = __riscv_vfmul_vf_f64m2(ab0, alpha_r, vl);
		ab1 = __riscv_vfmul_vf_f64m2(ab1, alpha_r, vl);
		ab2 = __riscv_vfmul_vf_f64m2(ab2, alpha_r, vl);
		ab3 = __riscv_vfmul_vf_f64m2(ab3, alpha_r, vl);
		ab4 = __riscv_vfmul_vf_f64m2(ab4, alpha_r, vl);
		ab5 = __riscv_vfmul_vf_f64m2(ab5, alpha_r, vl);
		ab6 = __riscv_vfmul_vf_f64m2(ab6, alpha_r, vl);
		ab7 = __riscv_vfmul_vf_f64m2(ab7, alpha_r, vl);
	}

	double *c0=cp, *c1=cp+cs_c, *c2=c1+cs_c, *c3=c2+cs_c;
	double *c4=c3+cs_c, *c5=c4+cs_c, *c6=c5+cs_c, *c7=c6+cs_c;

	if ( beta_r == 0.0 )
	{
		__riscv_vse64_v_f64m2(c0, ab0, vl); __riscv_vse64_v_f64m2(c1, ab1, vl);
		__riscv_vse64_v_f64m2(c2, ab2, vl); __riscv_vse64_v_f64m2(c3, ab3, vl);
		__riscv_vse64_v_f64m2(c4, ab4, vl); __riscv_vse64_v_f64m2(c5, ab5, vl);
		__riscv_vse64_v_f64m2(c6, ab6, vl); __riscv_vse64_v_f64m2(c7, ab7, vl);
	}
	else if ( beta_r == 1.0 )
	{
		ab0 = __riscv_vfadd_vv_f64m2(ab0, __riscv_vle64_v_f64m2(c0, vl), vl);
		ab1 = __riscv_vfadd_vv_f64m2(ab1, __riscv_vle64_v_f64m2(c1, vl), vl);
		ab2 = __riscv_vfadd_vv_f64m2(ab2, __riscv_vle64_v_f64m2(c2, vl), vl);
		ab3 = __riscv_vfadd_vv_f64m2(ab3, __riscv_vle64_v_f64m2(c3, vl), vl);
		ab4 = __riscv_vfadd_vv_f64m2(ab4, __riscv_vle64_v_f64m2(c4, vl), vl);
		ab5 = __riscv_vfadd_vv_f64m2(ab5, __riscv_vle64_v_f64m2(c5, vl), vl);
		ab6 = __riscv_vfadd_vv_f64m2(ab6, __riscv_vle64_v_f64m2(c6, vl), vl);
		ab7 = __riscv_vfadd_vv_f64m2(ab7, __riscv_vle64_v_f64m2(c7, vl), vl);
		__riscv_vse64_v_f64m2(c0, ab0, vl); __riscv_vse64_v_f64m2(c1, ab1, vl);
		__riscv_vse64_v_f64m2(c2, ab2, vl); __riscv_vse64_v_f64m2(c3, ab3, vl);
		__riscv_vse64_v_f64m2(c4, ab4, vl); __riscv_vse64_v_f64m2(c5, ab5, vl);
		__riscv_vse64_v_f64m2(c6, ab6, vl); __riscv_vse64_v_f64m2(c7, ab7, vl);
	}
	else
	{
		ab0 = __riscv_vfmacc_vf_f64m2(ab0, beta_r, __riscv_vle64_v_f64m2(c0, vl), vl);
		ab1 = __riscv_vfmacc_vf_f64m2(ab1, beta_r, __riscv_vle64_v_f64m2(c1, vl), vl);
		ab2 = __riscv_vfmacc_vf_f64m2(ab2, beta_r, __riscv_vle64_v_f64m2(c2, vl), vl);
		ab3 = __riscv_vfmacc_vf_f64m2(ab3, beta_r, __riscv_vle64_v_f64m2(c3, vl), vl);
		ab4 = __riscv_vfmacc_vf_f64m2(ab4, beta_r, __riscv_vle64_v_f64m2(c4, vl), vl);
		ab5 = __riscv_vfmacc_vf_f64m2(ab5, beta_r, __riscv_vle64_v_f64m2(c5, vl), vl);
		ab6 = __riscv_vfmacc_vf_f64m2(ab6, beta_r, __riscv_vle64_v_f64m2(c6, vl), vl);
		ab7 = __riscv_vfmacc_vf_f64m2(ab7, beta_r, __riscv_vle64_v_f64m2(c7, vl), vl);
		__riscv_vse64_v_f64m2(c0, ab0, vl); __riscv_vse64_v_f64m2(c1, ab1, vl);
		__riscv_vse64_v_f64m2(c2, ab2, vl); __riscv_vse64_v_f64m2(c3, ab3, vl);
		__riscv_vse64_v_f64m2(c4, ab4, vl); __riscv_vse64_v_f64m2(c5, ab5, vl);
		__riscv_vse64_v_f64m2(c6, ab6, vl); __riscv_vse64_v_f64m2(c7, ab7, vl);
	}
	GEMM_UKR_FLUSH_CT( d );
}
