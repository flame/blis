/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis2.h"

void bl2_strsm_l_4x4(
                      float* restrict a11,
                      float* restrict b11,
                      float* restrict bd11,
                      float* restrict c11, inc_t rs_c, inc_t cs_c
                    )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bl2_dtrsm_l_4x4(
                      double* restrict  a11,
                      double* restrict  b11,
                      double* restrict  bd11,
                      double* restrict  c11, inc_t rs_c, inc_t cs_c
                    )
{
	const  dim_t rs_a   = 1;
	const  dim_t cs_a   = 4;

	const  dim_t rs_b   = 4;
	const  dim_t cs_b   = 1;

	double beta00, beta01, beta02, beta03;
	double beta10, beta11, beta12, beta13;
	double beta20, beta21, beta22, beta23;
	double beta30, beta31, beta32, beta33;

	double alpha00;
	double alpha10, alpha11;
	double alpha20, alpha21, alpha22;
	double alpha30, alpha31, alpha32, alpha33;


	beta00 = *(b11 + 0*rs_b + 0*cs_b);
	beta01 = *(b11 + 0*rs_b + 1*cs_b);
	beta02 = *(b11 + 0*rs_b + 2*cs_b);
	beta03 = *(b11 + 0*rs_b + 3*cs_b);
	beta10 = *(b11 + 1*rs_b + 0*cs_b);
	beta11 = *(b11 + 1*rs_b + 1*cs_b);
	beta12 = *(b11 + 1*rs_b + 2*cs_b);
	beta13 = *(b11 + 1*rs_b + 3*cs_b);
	beta20 = *(b11 + 2*rs_b + 0*cs_b);
	beta21 = *(b11 + 2*rs_b + 1*cs_b);
	beta22 = *(b11 + 2*rs_b + 2*cs_b);
	beta23 = *(b11 + 2*rs_b + 3*cs_b);
	beta30 = *(b11 + 3*rs_b + 0*cs_b);
	beta31 = *(b11 + 3*rs_b + 1*cs_b);
	beta32 = *(b11 + 3*rs_b + 2*cs_b);
	beta33 = *(b11 + 3*rs_b + 3*cs_b);


	// iteration 0

	alpha00 = *(a11 + 0*rs_a + 0*cs_a);

	beta00 -= 0.0;
	beta01 -= 0.0;
	beta02 -= 0.0;
	beta03 -= 0.0;

	beta00 *= alpha00;
	beta01 *= alpha00;
	beta02 *= alpha00;
	beta03 *= alpha00;

	*(b11 + 0*rs_b + 0*cs_b) = beta00;
	*(b11 + 0*rs_b + 1*cs_b) = beta01;
	*(b11 + 0*rs_b + 2*cs_b) = beta02;
	*(b11 + 0*rs_b + 3*cs_b) = beta03;
	*(c11 + 0*rs_c + 0*cs_c) = beta00;
	*(c11 + 0*rs_c + 1*cs_c) = beta01;
	*(c11 + 0*rs_c + 2*cs_c) = beta02;
	*(c11 + 0*rs_c + 3*cs_c) = beta03;


	// iteration 1

	alpha10 = *(a11 + 1*rs_a + 0*cs_a);
	alpha11 = *(a11 + 1*rs_a + 1*cs_a);

	beta10 -= alpha10 * beta00;
	beta11 -= alpha10 * beta01;
	beta12 -= alpha10 * beta02;
	beta13 -= alpha10 * beta03;

	beta10 *= alpha11;
	beta11 *= alpha11;
	beta12 *= alpha11;
	beta13 *= alpha11;

	*(b11 + 1*rs_b + 0*cs_b) = beta10;
	*(b11 + 1*rs_b + 1*cs_b) = beta11;
	*(b11 + 1*rs_b + 2*cs_b) = beta12;
	*(b11 + 1*rs_b + 3*cs_b) = beta13;
	*(c11 + 1*rs_c + 0*cs_c) = beta10;
	*(c11 + 1*rs_c + 1*cs_c) = beta11;
	*(c11 + 1*rs_c + 2*cs_c) = beta12;
	*(c11 + 1*rs_c + 3*cs_c) = beta13;


	// iteration 2

	alpha20 = *(a11 + 2*rs_a + 0*cs_a);
	alpha21 = *(a11 + 2*rs_a + 1*cs_a);
	alpha22 = *(a11 + 2*rs_a + 2*cs_a);

	beta20 -= alpha20 * beta00 +
	          alpha21 * beta10;
	beta21 -= alpha20 * beta01 +
	          alpha21 * beta11;
	beta22 -= alpha20 * beta02 +
	          alpha21 * beta12;
	beta23 -= alpha20 * beta03 +
	          alpha21 * beta13;

	beta20 *= alpha22;
	beta21 *= alpha22;
	beta22 *= alpha22;
	beta23 *= alpha22;

	*(b11 + 2*rs_b + 0*cs_b) = beta20;
	*(b11 + 2*rs_b + 1*cs_b) = beta21;
	*(b11 + 2*rs_b + 2*cs_b) = beta22;
	*(b11 + 2*rs_b + 3*cs_b) = beta23;
	*(c11 + 2*rs_c + 0*cs_c) = beta20;
	*(c11 + 2*rs_c + 1*cs_c) = beta21;
	*(c11 + 2*rs_c + 2*cs_c) = beta22;
	*(c11 + 2*rs_c + 3*cs_c) = beta23;


	// iteration 3

	alpha30 = *(a11 + 3*rs_a + 0*cs_a);
	alpha31 = *(a11 + 3*rs_a + 1*cs_a);
	alpha32 = *(a11 + 3*rs_a + 2*cs_a);
	alpha33 = *(a11 + 3*rs_a + 3*cs_a);

	beta30 -= alpha30 * beta00 +
	          alpha31 * beta10 +
	          alpha32 * beta20;
	beta31 -= alpha30 * beta01 +
	          alpha31 * beta11 +
	          alpha32 * beta21;
	beta32 -= alpha30 * beta02 +
	          alpha31 * beta12 +
	          alpha32 * beta22;
	beta33 -= alpha30 * beta03 +
	          alpha31 * beta13 +
	          alpha32 * beta23;

	beta30 *= alpha33;
	beta31 *= alpha33;
	beta32 *= alpha33;
	beta33 *= alpha33;

	*(b11 + 3*rs_b + 0*cs_b) = beta30;
	*(b11 + 3*rs_b + 1*cs_b) = beta31;
	*(b11 + 3*rs_b + 2*cs_b) = beta32;
	*(b11 + 3*rs_b + 3*cs_b) = beta33;
	*(c11 + 3*rs_c + 0*cs_c) = beta30;
	*(c11 + 3*rs_c + 1*cs_c) = beta31;
	*(c11 + 3*rs_c + 2*cs_c) = beta32;
	*(c11 + 3*rs_c + 3*cs_c) = beta33;
}

void bl2_ctrsm_l_4x4(
                      scomplex* restrict a11,
                      scomplex* restrict b11,
                      scomplex* restrict bd11,
                      scomplex* restrict c11, inc_t rs_c, inc_t cs_c
                    )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bl2_ztrsm_l_4x4(
                      dcomplex* restrict a11,
                      dcomplex* restrict b11,
                      dcomplex* restrict bd11,
                      dcomplex* restrict c11, inc_t rs_c, inc_t cs_c
                    )
{
	bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

