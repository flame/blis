/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"
#include "immintrin.h"

typedef union{
	__m256d v;
	double d[4] __attribute__((aligned(64)));
}vec;

typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/**
 * bli_pre_hemv_lower_8x8 is a helper function which computes
 * "y = y + alpha * a * x"
 * dotxf and axpyf of triangular matrix with vector
 * for lower triangular matrix cases.
 * Computes 8 elements of Y vector by dot product
 * of 8 elements of x vector with 8x8 tile of A matrix
 * and axpy computation of each x vector elements with
 * each column of 8x8 A matrix tile.

*/
void bli_pre_hemv_8x8(double *a, double *x, double *y, double *alpha,
		dim_t cs_a, dim_t rs_a)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
	__m256d ymm10, ymm11, ymm12;
	double alpha_chi[8] = {0};
	/*Broadcast alpha*/
	ymm9 = _mm256_broadcast_sd(alpha);

	/**
	 * Scaling vector x with alpha
	 * to gather alpha_chi elements
	 * arranged in one buffer.
	 */
        ymm10 = _mm256_loadu_pd(x);
        ymm11 = _mm256_loadu_pd(x + 4);
        ymm10 = _mm256_mul_pd(ymm9, ymm10);
        ymm11 = _mm256_mul_pd(ymm9, ymm11);
        _mm256_storeu_pd(alpha_chi, ymm10);
        _mm256_storeu_pd(alpha_chi + 4, ymm11);

	/*Load y vector*/
	ymm10 = _mm256_loadu_pd(y);
	ymm11 = _mm256_loadu_pd(y + 4);

	//Col 0 computation
	/*Broadcasts chi and multiplies with alpha to get alpha chi*/
	ymm12 = _mm256_broadcast_sd(alpha_chi);
	/*Load first column of A matrix*/
	ymm0 = _mm256_loadu_pd(a);
	ymm1 = _mm256_loadu_pd(a + 4);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 1 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 1);
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 2nd column, packing to be done
	 * as shown below for ymm0:
	 * col-0 col-1
	 *  ---  ---
	     x    x
            ---   x
            ---   x
	 */
	ymm3 = _mm256_broadcast_sd(a + 1);
	ymm0 = _mm256_loadu_pd(a + cs_a * 1);
	ymm0 = _mm256_blend_pd(ymm0, ymm3, 0x1);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 1);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 2 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 2);
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 3rd column, packing to be done
	 * as shown below for ymm0:
	 * col-0 col-1 col-2
	 *  ---  ---  ---
	     x    x   ---
            ---  ---   x
            ---  ---   x
	 */
	ymm3 = _mm256_broadcast_sd(a + 2);
	ymm4 = _mm256_broadcast_sd(a + 2 + cs_a);
	ymm0 = _mm256_loadu_pd(a + cs_a * 2);
	ymm0 = _mm256_blend_pd(ymm0, ymm3, 0x1);
	ymm0 = _mm256_blend_pd(ymm0, ymm4, 0x2);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 2);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 3 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 3);
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-0 col-1 col-2 col-3
	 *  ---  ---  ---  ---
	     x    x    x   ---
            ---  ---  ---   x
	 */
	ymm3 = _mm256_broadcast_sd(a + 3);
	ymm4 = _mm256_broadcast_sd(a + 3 + cs_a);
	ymm5 = _mm256_broadcast_sd(a + 3 + cs_a * 2);
	ymm0 = _mm256_loadu_pd(a + cs_a * 3);
	ymm0 = _mm256_blend_pd(ymm0, ymm3, 0x1);
	ymm0 = _mm256_blend_pd(ymm0, ymm4, 0x2);
	ymm0 = _mm256_blend_pd(ymm0, ymm5, 0x4);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 3);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	/**
	 * Transpose 4x4 tile of matrix A,
	 * for remainder column computation.
	 */
	ymm0 = _mm256_loadu_pd(a+4 + cs_a * 0);
        ymm1 = _mm256_loadu_pd(a+4 + cs_a * 1);
        ymm2 = _mm256_loadu_pd(a+4 + cs_a * 2);
        ymm3 = _mm256_loadu_pd(a+4 + cs_a * 3);

        ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
        ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
        ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20); //Transposed col 1
        ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31); //Transposed col 3
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
        ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20); //Transposed col 2
        ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31); //Transposed col 4

	//Col 4 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 4);
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-0 col-1 col-2 col-3 col-4
	 *  ---  ---  ---  ---  ---
	     x    x    x    x   ---
            ---  ---  ---  ---  ---
            ---  ---  ---  ---  ---
	 */
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 4);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm6, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 5 computation
	/**
	 * Packs the data in similar manner as shown
	 * for col 0-4 computation, along with
	 * packing all 5th elements from col 0 - 4
	 * in other ymm register.
	 * col-4 col-5
	 *  ---  ---
	     x    x
            ---   x
            ---   x

	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 5);
	ymm3 = _mm256_broadcast_sd(a + 5 + cs_a * 4);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 5);
	ymm1 = _mm256_blend_pd(ymm1, ymm3, 0x1);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm7, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 6 computation
	/**
	 * Packs the data in similar manner as shown
	 * for col 0-4 computation, along with
	 * packing all 6th elements from col 0 - 4
	 * in other ymm register.
	 * col-4 col-5 col-6
	 *  ---  ---  ---
	     x    x   ---
            ---  ---   x
            ---  ---   x
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 6);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 6);
	ymm3 = _mm256_broadcast_sd(a + 6 + cs_a * 4);
	ymm4 = _mm256_broadcast_sd(a + 6 + cs_a * 5);
	ymm1 = _mm256_blend_pd(ymm1, ymm3, 0x1);
	ymm1 = _mm256_blend_pd(ymm1, ymm4, 0x2);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm8, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 7 computation
	/**
	 * Packs the data in similar manner as shown
	 * for col 0-4 computation, along with
	 * packing all 7th elements from col 0 - 4
	 * in other ymm register.
	 * col-4 col-5 col-6 col-7
	 *  ---  ---  ---  ---
	     x    x    x   ---
            ---  ---  ---   x
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 7);
	ymm1 = _mm256_loadu_pd(a + 4 + cs_a * 7);
	ymm3 = _mm256_broadcast_sd(a + 7 + cs_a * 4);
	ymm4 = _mm256_broadcast_sd(a + 7 + cs_a * 5);
	ymm5 = _mm256_broadcast_sd(a + 7 + cs_a * 6);
	ymm1 = _mm256_blend_pd(ymm1, ymm3, 0x1);
	ymm1 = _mm256_blend_pd(ymm1, ymm4, 0x2);
	ymm1 = _mm256_blend_pd(ymm1, ymm5, 0x4);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm9, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	/**
	 * Computed result of vector y is available in ymm10, ymm11.
	 * Storing the result back from ymm register into y vector for
	 * further computaion.
	 */
	_mm256_storeu_pd(y, ymm10);
	_mm256_storeu_pd(y + 4, ymm11);
}


/**
 * bli_post_hemv_lower_8x8 is a helper function which computes
 * "y = y + alpha * a * x"
 * dotxf and axpyf of triangular matrix with vector
 * for upper triangular matrix cases.
 * Computes 8 elements of Y vector by dot product
 * of 8 elements of x vector with 8x8 tile of A matrix
 * and axpy computation of each x vector elements with
 * each column of 8x8 A matrix tile.
*/
void bli_post_hemv_8x8(double *a, double *x, double *y, double *alpha,
		dim_t cs_a, dim_t rs_a)
{
	__m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
	__m256d ymm10, ymm11, ymm12;
	double alpha_chi[8] = {0};

	ymm9 = _mm256_broadcast_sd(alpha);

        ymm10 = _mm256_loadu_pd(x);
        ymm11 = _mm256_loadu_pd(x + 4);
        ymm10 = _mm256_mul_pd(ymm9, ymm10);
        ymm11 = _mm256_mul_pd(ymm9, ymm11);
        _mm256_storeu_pd(alpha_chi, ymm10);
        _mm256_storeu_pd(alpha_chi + 4, ymm11);

	ymm10 = _mm256_loadu_pd(y);
	ymm11 = _mm256_loadu_pd(y + 4);

        ymm0 = _mm256_loadu_pd(a + cs_a * 4);
        ymm1 = _mm256_loadu_pd(a + cs_a * 5);
        ymm2 = _mm256_loadu_pd(a + cs_a * 6);
        ymm3 = _mm256_loadu_pd(a + cs_a * 7);

        ymm4 = _mm256_unpacklo_pd(ymm0, ymm1);
        ymm5 = _mm256_unpacklo_pd(ymm2, ymm3);
        ymm6 = _mm256_permute2f128_pd(ymm4,ymm5,0x20);
        ymm8 = _mm256_permute2f128_pd(ymm4,ymm5,0x31);
        ymm0 = _mm256_unpackhi_pd(ymm0, ymm1);
        ymm1 = _mm256_unpackhi_pd(ymm2, ymm3);
        ymm7 = _mm256_permute2f128_pd(ymm0,ymm1,0x20);
        ymm9 = _mm256_permute2f128_pd(ymm0,ymm1,0x31);

	//Col 0 computation
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-0 col-1 col-2 col-3
	 *   x    x    x   x
            ---
	    ---
	    ---
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi);
	ymm0 = _mm256_loadu_pd(a);
	ymm1 = _mm256_broadcast_sd(a + cs_a * 1);
	ymm2 = _mm256_broadcast_sd(a + cs_a * 2);
	ymm3 = _mm256_broadcast_sd(a + cs_a * 3);
	ymm0 = _mm256_blend_pd(ymm0, ymm1, 0x2);
	ymm0 = _mm256_blend_pd(ymm0, ymm2, 0x4);
	ymm0 = _mm256_blend_pd(ymm0, ymm3, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm6, ymm11);

	//Col 1 computation
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-1 col-2 col-3
	 *   x    x    x
             x
	    ---
	    ---
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 1);
	ymm0 = _mm256_loadu_pd(a + cs_a * 1);
	ymm2 = _mm256_broadcast_sd(a + cs_a * 2 + 1);
	ymm3 = _mm256_broadcast_sd(a + cs_a * 3 + 1);
	ymm0 = _mm256_blend_pd(ymm0, ymm2, 0x4);
	ymm0 = _mm256_blend_pd(ymm0, ymm3, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm7, ymm11);

	//Col 2 computation
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-2 col-3
	 *   x    x
             x
	     x
	    ---
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 2);
	ymm0 = _mm256_loadu_pd(a + cs_a * 2);
	ymm2 = _mm256_broadcast_sd(a + cs_a * 3 + 2);
	ymm0 = _mm256_blend_pd(ymm0, ymm2, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm8, ymm11);

	//Col 3 computation
	/**
	 * pack the data in following manner into ymm register
	 * Since it is computing 4rd column, packing to be done
	 * as shown below for ymm0:
	 * col-3
	 *   x
             x
	     x
	     x
	 */
	ymm12 = _mm256_broadcast_sd(alpha_chi + 3);
	ymm0 = _mm256_loadu_pd(a + cs_a * 3);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm9, ymm11);

	//Col 4 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 4);
	ymm0 = _mm256_loadu_pd(a + cs_a * 4);
	ymm1 = _mm256_loadu_pd(a + cs_a * 4 + 4);
	ymm4 = _mm256_broadcast_sd(a + cs_a * 5 + 4);
	ymm5 = _mm256_broadcast_sd(a + cs_a * 6 + 4);
	ymm6 = _mm256_broadcast_sd(a + cs_a * 7 + 4);
	ymm1 = _mm256_blend_pd(ymm1, ymm4, 0x2);
	ymm1 = _mm256_blend_pd(ymm1, ymm5, 0x4);
	ymm1 = _mm256_blend_pd(ymm1, ymm6, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 5 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 5);
	ymm0 = _mm256_loadu_pd(a + cs_a * 5);
	ymm1 = _mm256_loadu_pd(a + cs_a * 5 + 4);
	ymm5 = _mm256_broadcast_sd(a + cs_a * 6 + 5);
	ymm6 = _mm256_broadcast_sd(a + cs_a * 7 + 5);
	ymm1 = _mm256_blend_pd(ymm1, ymm5, 0x4);
	ymm1 = _mm256_blend_pd(ymm1, ymm6, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 6 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 6);
	ymm0 = _mm256_loadu_pd(a + cs_a * 6);
	ymm1 = _mm256_loadu_pd(a + cs_a * 6 + 4);
	ymm6 = _mm256_broadcast_sd(a + cs_a * 7 + 6);
	ymm1 = _mm256_blend_pd(ymm1, ymm6, 0x8);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	//Col 7 computation
	ymm12 = _mm256_broadcast_sd(alpha_chi + 7);
	ymm0 = _mm256_loadu_pd(a + cs_a * 7);
	ymm1 = _mm256_loadu_pd(a + cs_a * 7 + 4);
	ymm10 = _mm256_fmadd_pd(ymm12, ymm0, ymm10);
	ymm11 = _mm256_fmadd_pd(ymm12, ymm1, ymm11);

	/**
	 * Computed result of vector y is available in ymm10, ymm11.
	 * Storing the result back from ymm register into y vector for
	 * further computaion.
	 */
	_mm256_storeu_pd(y, ymm10);
	_mm256_storeu_pd(y + 4, ymm11);
}


/**
 * ddotxaxpyf kernel performs dot and apxy function all togather
 * on a tile of 4x8 size.
 * x_trsv holds 4 elements of vector x, a_tile[0-7] holds
 * 4x8 tile of A matrix.
 * Following equations are solved in a way represented
 * y1 = y1 + alpha * A21' * x2;  (dotxf)
   y2 = y2 + alpha * A21  * x1;  (axpyf)

 *               B1    B2   B3    B4             B5   B6   B7   B8
 *               (broadcast elements of [x*alpha] vector)
 * tile          0     1     2      3             4     5     6     7
 * x_trsv[0]   A00   A01   A02   A03 => rho0  | A04   A05   A06   A07 =>  rho4
 * x_trsv[1]   A10   A11   A12   A13 => rho1  | A14   A15   A16   A17 =>  rho5
 * x_trsv[2]   A20   A21   A22   A23 => rho2  | A24   A25   A26   A27 =>  rho6
 * x_trsv[3]   A30   A31   A32   A33 => rho3  | A34   A35   A36   A37 =>  rho7
                ||    ||    ||    ||             ||    ||    ||    ||
	        \/    \/    \/    \/             \/    \/    \/    \/
	        +=    +=    +=    +=             +=    +=    +=    +=
             z_vec  z_vec  z_vec z_vec         z_vec  z_vec z_vec  z_vec
 *
 *
 */
void bli_ddotxaxpyf_zen_int_8
(
 conj_t           conjat,
 conj_t           conja,
 conj_t           conjw,
 conj_t           conjx,
 dim_t            m,
 dim_t            b_n,
 double*  restrict alpha,
 double*  restrict a, inc_t inca, inc_t lda,
 double*  restrict w, inc_t incw,
 double*  restrict x, inc_t incx,
 double*  restrict beta,
 double*  restrict y, inc_t incy,
 double*  restrict z, inc_t incz,
 cntx_t* restrict cntx
 )
{
	/* A is m x n.                  */
	/* y = beta * y + alpha * A^T w; */
	/* z =        z + alpha * A   x; */
	if ( ( bli_cpuid_is_avx2fma3_supported() == TRUE ) &&
	     (inca == 1) && (incw == 1) && (incx == 1)
	     && (incy == 1) && (incz == 1) && (b_n == 8) )
	{
		 __m256d r0, r1;
		 r0 = _mm256_setzero_pd();
		 r1 = _mm256_setzero_pd();

		/* If beta is zero, clear y. Otherwise, scale by beta. */
		if ( PASTEMAC(d,eq0)( *beta ) )
		{
			for ( dim_t i = 0; i < 8; ++i )
			{
				PASTEMAC(d,set0s)( y[i] );
			}
		}
		else
		{
			for ( dim_t i = 0; i < 8; ++i )
			{
				PASTEMAC(d,scals)( *beta, y[i] );
			}
		}

		/* If the vectors are empty or if alpha is zero, return early*/
		if ( bli_zero_dim1( m ) || PASTEMAC(d,eq0)( *alpha ) ) return;

		dim_t row = 0;
		dim_t iter = m/4;
		dim_t rem = m%4;
		if(iter)
		{
			vec x_trsv, x_hemvB1, x_hemvB2, x_hemvB3, x_hemvB4;
			vec  x_hemvB5, x_hemvB6, x_hemvB7, x_hemvB8;

			vec a_tile0, a_tile1, a_tile2, a_tile3;
			vec a_tile4, a_tile5, a_tile6, a_tile7;

			vec rho0, rho1, rho2, rho3;
			vec rho4, rho5, rho6, rho7;

			__m256d z_vec;

			/**
			 * Load [x vector * alpha], broadcast each element into
			 * different ymm registers. To perform axpyf operation
			 * with 4x8 tile of A matrix.
			 */

                        x_hemvB1.v = _mm256_set1_pd(x[0*incx] * (*alpha));
                        x_hemvB2.v = _mm256_set1_pd(x[1*incx] * (*alpha));
                        x_hemvB3.v = _mm256_set1_pd(x[2*incx] * (*alpha));
                        x_hemvB4.v = _mm256_set1_pd(x[3*incx] * (*alpha));

                        x_hemvB5.v = _mm256_set1_pd(x[4*incx] * (*alpha));
                        x_hemvB6.v = _mm256_set1_pd(x[5*incx] * (*alpha));
                        x_hemvB7.v = _mm256_set1_pd(x[6*incx] * (*alpha));
                        x_hemvB8.v = _mm256_set1_pd(x[7*incx] * (*alpha));

			/**
			 * clear rho register which holds result of
			 * fmadds for dotxf operation.
			 * Once micro tile is computed, horizontal addition
			 * of all rho's will provide us with the result of
			 * dotxf opereation.
			 */
			rho0.v = _mm256_setzero_pd();
			rho1.v = _mm256_setzero_pd();
			rho2.v = _mm256_setzero_pd();
			rho3.v = _mm256_setzero_pd();
			rho4.v = _mm256_setzero_pd();
			rho5.v = _mm256_setzero_pd();
			rho6.v = _mm256_setzero_pd();
			rho7.v = _mm256_setzero_pd();

			for(; (row + 3) < m; row+= 4)
			{
				a_tile0.v = _mm256_loadu_pd((double *)
						&a[row + 0 * lda] );
				a_tile1.v = _mm256_loadu_pd((double *)
						&a[row + 1 * lda] );
				a_tile2.v = _mm256_loadu_pd((double *)
						&a[row + 2 * lda] );
				a_tile3.v = _mm256_loadu_pd((double *)
						&a[row + 3 * lda] );
				a_tile4.v = _mm256_loadu_pd((double *)
						&a[row + 4 * lda] );
				a_tile5.v = _mm256_loadu_pd((double *)
						&a[row + 5 * lda] );
				a_tile6.v = _mm256_loadu_pd((double *)
						&a[row + 6 * lda] );
				a_tile7.v = _mm256_loadu_pd((double *)
						&a[row + 7 * lda] );

				x_trsv.v = _mm256_loadu_pd((double *) &w[row]);
				z_vec = _mm256_loadu_pd((double *) &z[row] );

				//dot product operation
				rho0.v = _mm256_fmadd_pd(a_tile0.v,
						x_trsv.v, rho0.v);
				rho4.v = _mm256_fmadd_pd(a_tile4.v,
						x_trsv.v, rho4.v);

				rho1.v = _mm256_fmadd_pd(a_tile1.v,
						x_trsv.v, rho1.v);
				rho5.v = _mm256_fmadd_pd(a_tile5.v,
						x_trsv.v, rho5.v);

				rho2.v = _mm256_fmadd_pd(a_tile2.v,
						x_trsv.v, rho2.v);
				rho6.v = _mm256_fmadd_pd(a_tile6.v,
						x_trsv.v, rho6.v);

				rho3.v = _mm256_fmadd_pd(a_tile3.v,
						x_trsv.v, rho3.v);
				rho7.v = _mm256_fmadd_pd(a_tile7.v,
						x_trsv.v, rho7.v);

				//axpy operation
				z_vec = _mm256_fmadd_pd(a_tile0.v,
						x_hemvB1.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile1.v,
						x_hemvB2.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile2.v,
						x_hemvB3.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile3.v,
						x_hemvB4.v, z_vec);

				z_vec = _mm256_fmadd_pd(a_tile4.v,
						x_hemvB5.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile5.v,
						x_hemvB6.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile6.v,
						x_hemvB7.v, z_vec);
				z_vec = _mm256_fmadd_pd(a_tile7.v,
						x_hemvB8.v, z_vec);

				_mm256_storeu_pd((double *)&z[row], z_vec);
			}
			/*Horizontal addition of rho's elements to compute
			 * the final dotxf result.
			 */
                        rho0.v = _mm256_hadd_pd( rho0.v, rho1.v );
                        rho2.v = _mm256_hadd_pd( rho2.v, rho3.v );
                        rho4.v = _mm256_hadd_pd( rho4.v, rho5.v );
                        rho6.v = _mm256_hadd_pd( rho6.v, rho7.v );

			{
				__m128d xmm0, xmm1;

				xmm0 = _mm256_extractf128_pd(rho0.v, 0);
				xmm1 = _mm256_extractf128_pd(rho0.v, 1);
				xmm0 = _mm_add_pd(xmm0, xmm1);
				r0 = _mm256_insertf128_pd(r0, xmm0, 0);

				xmm0 = _mm256_extractf128_pd(rho2.v, 0);
				xmm1 = _mm256_extractf128_pd(rho2.v, 1);
				xmm0 = _mm_add_pd(xmm0, xmm1);
				r0 = _mm256_insertf128_pd(r0, xmm0, 1);


				xmm0 = _mm256_extractf128_pd(rho4.v, 0);
				xmm1 = _mm256_extractf128_pd(rho4.v, 1);
				xmm0 = _mm_add_pd(xmm0, xmm1);
				r1 = _mm256_insertf128_pd(r1, xmm0, 0);

				xmm0 = _mm256_extractf128_pd(rho6.v, 0);
				xmm1 = _mm256_extractf128_pd(rho6.v, 1);
				xmm0 = _mm_add_pd(xmm0, xmm1);
				r1 = _mm256_insertf128_pd(r1, xmm0, 1);
			}
		}
		if(rem)
		{
			double r[ 8 ];
			double ax[ 8 ];
			/**
			 * Computed dot product computation needs
			 * to be brought into the r buffer for
			 * corner cases, so that remainder computation
			 * can be updated in r buffer.
			 */
			_mm256_storeu_pd((double *)r, r0);
                        _mm256_storeu_pd( (double *)(r + 4), r1);

			PRAGMA_SIMD
				for ( dim_t i = 0; i < 8; ++i )
				{
					PASTEMAC(d,scal2s)
						( *alpha, x[i], ax[i] );
				}

			PRAGMA_SIMD
				for ( dim_t p = row; p < m; ++p )
				{
					for ( dim_t i = 0; i < 8; ++i )
					{
						PASTEMAC(d,axpys)
							( a[p + i*lda],
							  w[p], r[i] );
						PASTEMAC(d,axpyjs)
							( ax[i],
							  a[p + i*lda], z[p] );
					}
				}
			/**
			 * Final dot product computation needs be
			 * loaded into registers, for getting
			 * scaled by Alpha and finally be stored
			 * back into output vector.
			 */
			r0 = _mm256_loadu_pd((double const *)r);
                        r1 = _mm256_loadu_pd((double const *)(r + 4));
		}

		/**
		 * Storing the computed result after being
		 * scaled by Alpha into output vector.
		 */
		{
			__m256d y0, y1, Alpha;
			y0 = _mm256_loadu_pd(y);
			y1 = _mm256_loadu_pd(y + 4);
			Alpha = _mm256_broadcast_sd(alpha);
			y0 = _mm256_fmadd_pd(Alpha, r0, y0);
			y1 = _mm256_fmadd_pd(Alpha, r1, y1);
			_mm256_storeu_pd(y, y0);
			_mm256_storeu_pd(y+4, y1);
		}
	}
	else
	{
		/* Query the context for the kernel function pointer. */
		const num_t dt     = PASTEMAC(d,type);
		PASTECH(d,dotxf_ker_ft) kfp_df 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
		PASTECH(d,axpyf_ker_ft) kfp_af 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx );

		kfp_df
			(
			 conjat,
			 conjw,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 w, incw,
			 beta,
			 y, incy,
			 cntx
			);

		kfp_af
			(
			 conja,
			 conjx,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 x, incx,
			 z, incz,
			 cntx
			);
	}
}

/**
 * zdotxaxpyf kernel performs dot and apxy function together.
 * y := conj(beta) * y + conj(alpha) * conj(A)^t * conj(w)      (dotxf)
 * z := z + alpha * conj(A) * conj(x)                           (axpyf)
 * where,
 *      A is an m x b matrix.
 *      w, z are vectors of length m.
 *      x, y are vectors of length b.
 *      alpha, beta are scalars
 */
void bli_zdotxaxpyf_zen_int_8
(
 conj_t              conjat,
 conj_t              conja,
 conj_t              conjw,
 conj_t              conjx,
 dim_t               m,
 dim_t               b_n,
 dcomplex*  restrict alpha,
 dcomplex*  restrict a, inc_t inca, inc_t lda,
 dcomplex*  restrict w, inc_t incw,
 dcomplex*  restrict x, inc_t incx,
 dcomplex*  restrict beta,
 dcomplex*  restrict y, inc_t incy,
 dcomplex*  restrict z, inc_t incz,
 cntx_t*	restrict cntx
 )
{
	// 	  A: m x b
	// w, z: m
	// x, y: b
	//
	// y = beta * y + alpha * A^T w;
	// z =        z + alpha * A   x;
	if ( ( bli_cpuid_is_avx2fma3_supported() == TRUE ) &&
	     ( inca == 1 ) && ( incw == 1 ) && ( incx == 1 )
	     && ( incy == 1 ) && ( incz == 1 ) && ( b_n == 4 ) )
	{
		// Temporary rho buffer holds computed dot product result
		dcomplex 	rho[ 4 ];

		// chi? variables to hold scaled scaler values from x vector
		dcomplex	chi0;
		dcomplex	chi1;
		dcomplex	chi2;
		dcomplex	chi3;

		// If beta is zero, clear y
		// Else, scale by beta
		if ( PASTEMAC(z,eq0)( *beta ) )
		{
			for ( dim_t i = 0; i < 4; ++i )
			{
				PASTEMAC(z,set0s)( y[i] );
			}
		}
		else
		{
			for ( dim_t i = 0; i < 4; ++i )
			{
				PASTEMAC(z,scals)( *beta, y[i] );
			}
		}

		// If the vectors are empty or if alpha is zero, return early
		if ( bli_zero_dim1( m ) || PASTEMAC(z,eq0)( *alpha ) ) return;

		// Initialize rho vector to 0
		for ( dim_t i = 0; i < 4; ++i ) PASTEMAC(z,set0s)( rho[i] );

		// Set conj use variable for dot operation
		conj_t conjdot_use = conjw;
		if ( bli_is_conj( conjat ) )
		{
			bli_toggle_conj( &conjdot_use );
		}

		// Set conj use variable for dotxf operation, scalar
		dim_t conjdotxf = 1;
		if ( bli_is_conj( conjdot_use ) )
		{
			conjdotxf = -1;
		}

		// Set conj use variable for axpyf operation, scalar
		dim_t conjaxpyf = 1;
		if ( bli_is_conj( conja ) )
		{
			conjaxpyf = -1;
		}

		// Store each element of x vector in a scalar and apply conjx
		if( bli_is_noconj( conjx ) )
		{
			chi0 = *( x + 0*incx );
			chi1 = *( x + 1*incx );
			chi2 = *( x + 2*incx );
			chi3 = *( x + 3*incx );
		}
		else
		{
			bli_zcopycjs( conjx, *( x + 0*incx ), chi0 );
			bli_zcopycjs( conjx, *( x + 1*incx ), chi1 );
			bli_zcopycjs( conjx, *( x + 2*incx ), chi2 );
			bli_zcopycjs( conjx, *( x + 3*incx ), chi3 );
		}

		// Scale each chi scalar by alpha
		bli_zscals( *alpha, chi0 );
		bli_zscals( *alpha, chi1 );
		bli_zscals( *alpha, chi2 );
		bli_zscals( *alpha, chi3 );

		dim_t row = 0;
		dim_t iter = m / 2;
		dim_t rem = m % 2;
		if (iter)
		{
			vec x0R, x1R, x2R, x3R;     // x?R holds real part of x[?]
			vec x0I, x1I, x2I, x3I;     // x?I hold real part of x[?]
			vec a_tile0, a_tile1;       // a_tile? holds columns of a
			vec temp1, temp2, temp3;    // temp? registers for intermediate op
			vec wR, wI;                 // holds real & imag components of w
			vec z_vec;                  // holds the z vector

			// rho? registers hold results of fmadds for dotxf operation
			vec rho0, rho1, rho2, rho3;
			vec rho4, rho5, rho6, rho7;

			// For final computation, based on conjdot_use
			// sign of imaginary component needs to be toggled
			__m256d no_conju = _mm256_setr_pd( -1,  1, -1,  1 );
			__m256d conju    = _mm256_setr_pd(  1, -1,  1, -1 );

			// Clear the temp registers
			temp1.v = _mm256_setzero_pd();
			temp2.v = _mm256_setzero_pd();
			temp3.v = _mm256_setzero_pd();

			// Clear rho registers
			// Once micro tile is computed, horizontal addition
			// of all rho's will provide us with the result of
			// dotxf opereation
			rho0.v = _mm256_setzero_pd();
			rho1.v = _mm256_setzero_pd();
			rho2.v = _mm256_setzero_pd();
			rho3.v = _mm256_setzero_pd();
			rho4.v = _mm256_setzero_pd();
			rho5.v = _mm256_setzero_pd();
			rho6.v = _mm256_setzero_pd();
			rho7.v = _mm256_setzero_pd();

			// Broadcast real & imag parts of 4 elements of x
		 	// to perform axpyf operation with 4x8 tile of A
			x0R.v = _mm256_broadcast_sd( &chi0.real );	// real part of x0
			x0I.v = _mm256_broadcast_sd( &chi0.imag );	// imag part of x0
			x1R.v = _mm256_broadcast_sd( &chi1.real );	// real part of x1
			x1I.v = _mm256_broadcast_sd( &chi1.imag );	// imag part of x1
			x2R.v = _mm256_broadcast_sd( &chi2.real );	// real part of x2
			x2I.v = _mm256_broadcast_sd( &chi2.imag );	// imag part of x2
			x3R.v = _mm256_broadcast_sd( &chi3.real );	// real part of x3
			x3I.v = _mm256_broadcast_sd( &chi3.imag );	// imag part of x3

			for ( ; ( row + 1 ) < m; row += 2)
			{
				// Load first two columns of A
				// a_tile0.v -> a00R a00I a10R a10I
				// a_tile1.v -> a01R a01I a11R a11I
				a_tile0.v = _mm256_loadu_pd( (double *)&a[row + 0 * lda] );
				a_tile1.v = _mm256_loadu_pd( (double *)&a[row + 1 * lda] );

				temp1.v = _mm256_mul_pd( a_tile0.v, x0R.v );
				temp2.v = _mm256_mul_pd( a_tile0.v, x0I.v );

				temp1.v = _mm256_fmadd_pd( a_tile1.v, x1R.v, temp1.v );
				temp2.v = _mm256_fmadd_pd( a_tile1.v, x1I.v, temp2.v );

				// Load w vector
				// wR.v                 -> w0R w0I w1R w1I
				// wI.v ( shuf wR.v )   -> w0I w0I w1I w1I
				// wR.v ( shuf wR.v )   -> w0R w0R w1R w1R
				wR.v =   _mm256_loadu_pd( (double *)&w[row] );
				wI.v = _mm256_permute_pd( wR.v, 15 );
				wR.v = _mm256_permute_pd( wR.v, 0 );

				rho0.v = _mm256_fmadd_pd( a_tile0.v, wR.v, rho0.v);
				rho4.v = _mm256_fmadd_pd( a_tile0.v, wI.v, rho4.v);

				rho1.v = _mm256_fmadd_pd( a_tile1.v, wR.v, rho1.v);
				rho5.v = _mm256_fmadd_pd( a_tile1.v, wI.v, rho5.v);

				// Load 3rd and 4th columns of A
				// a_tile0.v -> a20R a20I a30R a30I
				// a_tile1.v -> a21R a21I a31R a31I
				a_tile0.v = _mm256_loadu_pd( (double *)&a[row + 2 * lda] );
				a_tile1.v = _mm256_loadu_pd( (double *)&a[row + 3 * lda] );

				temp1.v = _mm256_fmadd_pd( a_tile0.v, x2R.v, temp1.v );
				temp2.v = _mm256_fmadd_pd( a_tile0.v, x2I.v, temp2.v );

				temp1.v = _mm256_fmadd_pd( a_tile1.v, x3R.v, temp1.v );
				temp2.v = _mm256_fmadd_pd( a_tile1.v, x3I.v, temp2.v );

				rho2.v = _mm256_fmadd_pd( a_tile0.v, wR.v, rho2.v);
				rho6.v = _mm256_fmadd_pd( a_tile0.v, wI.v, rho6.v);

				rho3.v = _mm256_fmadd_pd( a_tile1.v, wR.v, rho3.v);
				rho7.v = _mm256_fmadd_pd( a_tile1.v, wI.v, rho7.v);

				// Load z vector
				z_vec.v = _mm256_loadu_pd( (double *)&z[row] );

				// Permute the result and alternatively add-sub final values
				if( bli_is_noconj( conja ) )
				{
					temp2.v = _mm256_permute_pd(temp2.v, 5);
					temp3.v = _mm256_addsub_pd(temp1.v, temp2.v);
				}
				else
				{
					temp1.v = _mm256_permute_pd( temp1.v, 5 );
					temp3.v = _mm256_addsub_pd( temp2.v, temp1.v );
					temp3.v = _mm256_permute_pd( temp3.v, 5 );
				}

				// Add & store result to z_vec
				z_vec.v = _mm256_add_pd( temp3.v, z_vec.v );
				_mm256_storeu_pd( (double *)&z[row], z_vec.v );
			}

			// Swapping position of real and imag component
			// for horizontal addition to get the final
			// dot product computation
			// rho register are holding computation which needs
			// to be arranged in following manner.
			// a0R * x0I | a0I * x0I | a1R * x1I | a1I * x1R
			//                      ||
			//                      \/
			// a0I * x0I | a0R * x0I | a1I * x1R | a1R * x1I

			rho4.v = _mm256_permute_pd(rho4.v, 0x05);
			rho5.v = _mm256_permute_pd(rho5.v, 0x05);
			rho6.v = _mm256_permute_pd(rho6.v, 0x05);
			rho7.v = _mm256_permute_pd(rho7.v, 0x05);

			// Negating imaginary part for computing
			// the final result of dcomplex multiplication
			if ( bli_is_noconj( conjdot_use ) )
			{
				rho4.v = _mm256_mul_pd(rho4.v, no_conju);
				rho5.v = _mm256_mul_pd(rho5.v, no_conju);
				rho6.v = _mm256_mul_pd(rho6.v, no_conju);
				rho7.v = _mm256_mul_pd(rho7.v, no_conju);
			}
			else
			{
				rho4.v = _mm256_mul_pd(rho4.v, conju);
				rho5.v = _mm256_mul_pd(rho5.v, conju);
				rho6.v = _mm256_mul_pd(rho6.v, conju);
				rho7.v = _mm256_mul_pd(rho7.v, conju);
			}

			rho0.v = _mm256_add_pd(rho0.v, rho4.v);
			rho1.v = _mm256_add_pd(rho1.v, rho5.v);
			rho2.v = _mm256_add_pd(rho2.v, rho6.v);
			rho3.v = _mm256_add_pd(rho3.v, rho7.v);

			// rho0 & rho1 hold final dot product
			// result of 4 dcomplex elements
			rho0.d[0] += rho0.d[2];
			rho0.d[1] += rho0.d[3];

			rho0.d[2] = rho1.d[0] + rho1.d[2];
			rho0.d[3] = rho1.d[1] + rho1.d[3];

			rho1.d[0] = rho2.d[0] + rho2.d[2];
			rho1.d[1] = rho2.d[1] + rho2.d[3];

			rho1.d[2] = rho3.d[0] + rho3.d[2];
			rho1.d[3] = rho3.d[1] + rho3.d[3];

			// Storing the computed dot product
			// in temp buffer rho for further computation.
			_mm256_storeu_pd( (double *)rho, rho0.v );
			_mm256_storeu_pd( (double *)(rho+2) , rho1.v );
		}

		// To handle the remaining cases
		if ( rem )
		{
			PRAGMA_SIMD
				for ( dim_t p = row; p < m; ++p )
				{
					const dcomplex a0c = a[p + 0 * lda];
					const dcomplex a1c = a[p + 1 * lda];
					const dcomplex a2c = a[p + 2 * lda];
					const dcomplex a3c = a[p + 3 * lda];

					// dot
					dcomplex r0c = rho[0];
					dcomplex r1c = rho[1];
					dcomplex r2c = rho[2];
					dcomplex r3c = rho[3];

					dcomplex w0c = w[p];

					r0c.real += a0c.real * w0c.real - a0c.imag * w0c.imag
					            * conjdotxf;
					r0c.imag += a0c.imag * w0c.real + a0c.real * w0c.imag
					            * conjdotxf;
					r1c.real += a1c.real * w0c.real - a1c.imag * w0c.imag
					            * conjdotxf;
					r1c.imag += a1c.imag * w0c.real + a1c.real * w0c.imag
					            * conjdotxf;
					r2c.real += a2c.real * w0c.real - a2c.imag * w0c.imag
					            * conjdotxf;
					r2c.imag += a2c.imag * w0c.real + a2c.real * w0c.imag
					            * conjdotxf;
					r3c.real += a3c.real * w0c.real - a3c.imag * w0c.imag
					            * conjdotxf;
					r3c.imag += a3c.imag * w0c.real + a3c.real * w0c.imag
					            * conjdotxf;

					rho[0] = r0c;
					rho[1] = r1c;
					rho[2] = r2c;
					rho[3] = r3c;

					// axpy
					dcomplex z0c = z[p];

					z0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag
					            * conjaxpyf;
					z0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag
					            * conjaxpyf;
					z0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag
					            * conjaxpyf;
					z0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag
					            * conjaxpyf;
					z0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag
					            * conjaxpyf;
					z0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag
					            * conjaxpyf;
					z0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag
					            * conjaxpyf;
					z0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag
					            * conjaxpyf;

					z[p] = z0c;
				}
		}

		// Conjugating the final result if conjat
		if ( bli_is_conj( conjat ) )
		{
			for ( dim_t i = 0; i < 4; ++i )
			{
				PASTEMAC(z,conjs)( rho[i] );
			}
		}

		// Scaling the dot product result with alpha
		// and adding the result to vector y
		for ( dim_t i = 0; i < 4; ++i )
		{
			PASTEMAC(z,axpys)( *alpha, rho[i], y[i] );
		}
	}
	else
	{
		// For non-unit increments
		/* Query the context for the kernel function pointer. */
		const num_t dt     = PASTEMAC(z,type);
		PASTECH(z,dotxf_ker_ft) kfp_df 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
		PASTECH(z,axpyf_ker_ft) kfp_af 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx );

		kfp_df
			(
			 conjat,
			 conjw,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 w, incw,
			 beta,
			 y, incy,
			 cntx
			);

		kfp_af
			(
			 conja,
			 conjx,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 x, incx,
			 z, incz,
			 cntx
			);
	}
}

/**
 * cdotxaxpyf kernel performs dot and apxy function together.
 * y := conj(beta) * y + conj(alpha) * conj(A)^t * conj(w)      (dotxf)
 * z := z + alpha * conj(A) * conj(x)                           (axpyf)
 * where,
 *      A is an m x b matrix.
 *      w, z are vectors of length m.
 *      x, y are vectors of length b.
 *      alpha, beta are scalars
 */
void bli_cdotxaxpyf_zen_int_8
(
 conj_t              conjat,
 conj_t              conja,
 conj_t              conjw,
 conj_t              conjx,
 dim_t               m,
 dim_t               b_n,
 scomplex*  restrict alpha,
 scomplex*  restrict a, inc_t inca, inc_t lda,
 scomplex*  restrict w, inc_t incw, 
 scomplex*  restrict x, inc_t incx,
 scomplex*  restrict beta,
 scomplex*  restrict y, inc_t incy,
 scomplex*  restrict z, inc_t incz,
 cntx_t*	restrict cntx
 )
{
	//    A: m x b
	// w, z: m
	// x, y: b
	//
	// y = beta * y + alpha * A^T w;
	// z =        z + alpha * A   x;
	if ( ( bli_cpuid_is_avx2fma3_supported() == TRUE ) &&
	     ( inca == 1 ) && ( incw == 1 ) && ( incx == 1 )
	     && ( incy == 1 ) && ( incz == 1 ) && ( b_n == 4 ) )
	{
		// Temporary rho buffer holds computed dot product result
		scomplex rho[ 4 ];

		// chi? variables to hold scaled scaler values from x vector
		scomplex    chi0;
		scomplex    chi1;
		scomplex    chi2;
		scomplex    chi3;

		// If beta is zero, clear y
		// Else, scale by beta
		if ( PASTEMAC(c,eq0)( *beta ) )
		{
			for ( dim_t i = 0; i < 4; ++i )
			{
				PASTEMAC(c,set0s)( y[i] );
			}
		}
		else
		{
			for ( dim_t i = 0; i < 4; ++i )
			{
				PASTEMAC(c,scals)( *beta, y[i] );
			}
		}

		// If the vectors are empty or if alpha is zero, return early
		if ( bli_zero_dim1( m ) || PASTEMAC(c,eq0)( *alpha ) ) return;

		// Initialize rho vector to 0
		for ( dim_t i = 0; i < 4; ++i ) PASTEMAC(c,set0s)( rho[i] );

		// Set conj use variable for dot operation
		conj_t conjdot_use = conjw;
		if ( bli_is_conj( conjat ) )
		{
			bli_toggle_conj( &conjdot_use );
		}

		// Set conj use variable for dotxf operation, scalar
		dim_t conjdotxf = 1;
		if ( bli_is_conj( conjdot_use ) )
		{
			conjdotxf = -1;
		}

		// Set conj use variable for axpyf operation, scalar
		dim_t conjaxpyf = 1;
		if ( bli_is_conj( conja ) )
		{
			conjaxpyf = -1;
		}

		// Store each element of x vector in a scalar and apply conjx
		if( bli_is_noconj( conjx ) )
		{
			chi0 = *( x + 0*incx );
			chi1 = *( x + 1*incx );
			chi2 = *( x + 2*incx );
			chi3 = *( x + 3*incx );
		}
		else
		{
			bli_ccopycjs( conjx, *( x + 0*incx ), chi0 );
			bli_ccopycjs( conjx, *( x + 1*incx ), chi1 );
			bli_ccopycjs( conjx, *( x + 2*incx ), chi2 );
			bli_ccopycjs( conjx, *( x + 3*incx ), chi3 );
		}

		// Scale each chi scalar by alpha
		bli_cscals( *alpha, chi0 );
		bli_cscals( *alpha, chi1 );
		bli_cscals( *alpha, chi2 );
		bli_cscals( *alpha, chi3 );

		dim_t i = 0;
		dim_t iter = m / 4;
		dim_t rem = m % 4;
		if (iter)
		{
			v8sf_t x0R, x1R, x2R, x3R;     // x?R holds real part of x[?]
			v8sf_t x0I, x1I, x2I, x3I;     // x?I hold real part of x[?]
			v8sf_t a_tile0, a_tile1;       // a_tile? holds columns of a
			v8sf_t temp1, temp2, temp3;    // temp? registers for intermediate op
			v8sf_t wR, wI;                 // holds real & imag components of w
			v8sf_t z_vec;                  // holds the z vector

			// For final computation, based on conjdot_use
			// sign of imaginary component needs to be toggled
			__m256 no_conju = _mm256_setr_ps( -1,  1, -1,  1, -1,  1, -1,  1 );
			__m256 conju    = _mm256_setr_ps(  1, -1,  1, -1,  1, -1,  1, -1 );

			// Clear the temp registers
			temp1.v = _mm256_setzero_ps();
			temp2.v = _mm256_setzero_ps();
			temp3.v = _mm256_setzero_ps();

			// Clear rho registers
			// Once micro tile is computed, horizontal addition
			// of all rho's will provide us with the result of
			// dotxf opereation
			__m256 rho0v; rho0v = _mm256_setzero_ps();
			__m256 rho1v; rho1v = _mm256_setzero_ps();
			__m256 rho2v; rho2v = _mm256_setzero_ps();
			__m256 rho3v; rho3v = _mm256_setzero_ps();

			__m256 rho4v; rho4v = _mm256_setzero_ps();
			__m256 rho5v; rho5v = _mm256_setzero_ps();
			__m256 rho6v; rho6v = _mm256_setzero_ps();
			__m256 rho7v; rho7v = _mm256_setzero_ps();

			// Broadcast real & imag parts of 4 elements of x
		 	// to perform axpyf operation with 4x8 tile of A
			x0R.v = _mm256_broadcast_ss( &chi0.real );	// real part of x0
			x0I.v = _mm256_broadcast_ss( &chi0.imag );	// imag part of x0
			x1R.v = _mm256_broadcast_ss( &chi1.real );	// real part of x1
			x1I.v = _mm256_broadcast_ss( &chi1.imag );	// imag part of x1
			x2R.v = _mm256_broadcast_ss( &chi2.real );	// real part of x2
			x2I.v = _mm256_broadcast_ss( &chi2.imag );	// imag part of x2
			x3R.v = _mm256_broadcast_ss( &chi3.real );	// real part of x3
			x3I.v = _mm256_broadcast_ss( &chi3.imag );	// imag part of x3

			for ( ; ( i + 3 ) < m; i += 4)
			{
				// Load first two columns of A
				// a_tile0.v -> a00R a00I a10R a10I a20R a20I a30R a30I
				// a_tile1.v -> a01R a01I a11R a11I a21R a21I a31R a31I
				a_tile0.v = _mm256_loadu_ps( (float *)&a[i + 0 * lda] );
				a_tile1.v = _mm256_loadu_ps( (float *)&a[i + 1 * lda] );

				temp1.v = _mm256_mul_ps( a_tile0.v, x0R.v );
				temp2.v = _mm256_mul_ps( a_tile0.v, x0I.v );

				temp1.v = _mm256_fmadd_ps( a_tile1.v, x1R.v, temp1.v );
				temp2.v = _mm256_fmadd_ps( a_tile1.v, x1I.v, temp2.v );

				// Load w vector
				// wR.v                 -> w0R w0I w1R w1I w2R w2I w3R w3I
				// wI.v ( shuf wR.v )   -> w0I w0I w1I w1I w2I w2I w3I w3I
				// wR.v ( shuf wR.v )   -> w0R w0R w1R w1R w2R w2R w3R w3R
				wR.v = _mm256_loadu_ps( (float *) (w + i) );
				wI.v = _mm256_permute_ps( wR.v, 0xf5 );
				wR.v = _mm256_permute_ps( wR.v, 0xa0);

				rho0v = _mm256_fmadd_ps( a_tile0.v, wR.v, rho0v );
				rho4v = _mm256_fmadd_ps( a_tile0.v, wI.v, rho4v );

				rho1v = _mm256_fmadd_ps( a_tile1.v, wR.v, rho1v );
				rho5v = _mm256_fmadd_ps( a_tile1.v, wI.v, rho5v );

				// Load 3rd and 4th columns of A
				// a_tile0.v -> a20R a20I a30R a30I
				// a_tile1.v -> a21R a21I a31R a31I
				a_tile0.v = _mm256_loadu_ps( (float *)&a[i + 2 * lda] );
				a_tile1.v = _mm256_loadu_ps( (float *)&a[i + 3 * lda] );

				temp1.v = _mm256_fmadd_ps( a_tile0.v, x2R.v, temp1.v );
				temp2.v = _mm256_fmadd_ps( a_tile0.v, x2I.v, temp2.v );

				temp1.v = _mm256_fmadd_ps( a_tile1.v, x3R.v, temp1.v );
				temp2.v = _mm256_fmadd_ps( a_tile1.v, x3I.v, temp2.v );

				rho2v = _mm256_fmadd_ps( a_tile0.v, wR.v, rho2v );
				rho6v = _mm256_fmadd_ps( a_tile0.v, wI.v, rho6v );

				rho3v = _mm256_fmadd_ps( a_tile1.v, wR.v, rho3v );
				rho7v = _mm256_fmadd_ps( a_tile1.v, wI.v, rho7v );

				// Load z vector
				z_vec.v = _mm256_loadu_ps( (float *)&z[i] );

				// Permute the result and alternatively add-sub final values
				if( bli_is_noconj( conja ) )
				{
					temp2.v = _mm256_permute_ps(temp2.v, 0xB1);
					temp3.v =  _mm256_addsub_ps(temp1.v, temp2.v);
				}
				else
				{
					temp1.v = _mm256_permute_ps( temp1.v, 0xB1 );
					temp3.v =  _mm256_addsub_ps( temp2.v, temp1.v );
					temp3.v = _mm256_permute_ps( temp3.v, 0xB1 );
				}

				// Add & store result to z_vec
				z_vec.v = _mm256_add_ps( temp3.v, z_vec.v );
				_mm256_storeu_ps( (float *)&z[i], z_vec.v );
			}

			// Swapping position of real and imag component
			// for horizontal addition to get the final
			// dot product computation
			// rho register are holding computation which needs
			// to be arranged in following manner.
			// a0R * x0I | a0I * x0I | a1R * x1I | a1I * x1R | ...
			//                      ||
			//                      \/
			// a0I * x0I | a0R * x0I | a1I * x1R | a1R * x1I | ...

			rho4v = _mm256_permute_ps(rho4v, 0xb1);
			rho5v = _mm256_permute_ps(rho5v, 0xb1);
			rho6v = _mm256_permute_ps(rho6v, 0xb1);
			rho7v = _mm256_permute_ps(rho7v, 0xb1);

			// Negating imaginary part for computing
			// the final result of dcomplex multiplication
			if ( bli_is_noconj( conjdot_use ) )
			{
				rho4v = _mm256_mul_ps(rho4v, no_conju);
				rho5v = _mm256_mul_ps(rho5v, no_conju);
				rho6v = _mm256_mul_ps(rho6v, no_conju);
				rho7v = _mm256_mul_ps(rho7v, no_conju);
			}
			else
			{
				rho4v = _mm256_mul_ps(rho4v, conju);
				rho5v = _mm256_mul_ps(rho5v, conju);
				rho6v = _mm256_mul_ps(rho6v, conju);
				rho7v = _mm256_mul_ps(rho7v, conju);
			}

			rho0v = _mm256_add_ps(rho0v, rho4v);
			rho1v = _mm256_add_ps(rho1v, rho5v);
			rho2v = _mm256_add_ps(rho2v, rho6v);
			rho3v = _mm256_add_ps(rho3v, rho7v);

			// Horizontal addition of rho elements for computing final dotxf
			// and storing the results into rho buffer
			scomplex *ptr = (scomplex *)&rho0v;
			for(dim_t j = 0; j < 4; j++)
			{
				rho[0].real += ptr[j].real;
				rho[0].imag += ptr[j].imag;
			}
			ptr = (scomplex *)&rho1v;
			for(dim_t j = 0; j < 4; j++)
			{
				rho[1].real += ptr[j].real;
				rho[1].imag += ptr[j].imag;
			}
			ptr = (scomplex *)&rho2v;
			for(dim_t j = 0; j < 4; j++)
			{
				rho[2].real += ptr[j].real;
				rho[2].imag += ptr[j].imag;
			}
			ptr = (scomplex *)&rho3v;
			for(dim_t j = 0; j < 4; j++)
			{
				rho[3].real += ptr[j].real;
				rho[3].imag += ptr[j].imag;
			}
		}

		// To handle the remaining cases
		if ( rem )
		{
			PRAGMA_SIMD
			for ( dim_t p = i; p < m; ++p )
			{
				const scomplex a0c = a[p + 0 * lda];
				const scomplex a1c = a[p + 1 * lda];
				const scomplex a2c = a[p + 2 * lda];
				const scomplex a3c = a[p + 3 * lda];

				// dot
				scomplex r0c = rho[0];
				scomplex r1c = rho[1];
				scomplex r2c = rho[2];
				scomplex r3c = rho[3];

				scomplex w0c = w[p];

				r0c.real += a0c.real * w0c.real - a0c.imag * w0c.imag
				            * conjdotxf;
				r0c.imag += a0c.imag * w0c.real + a0c.real * w0c.imag
				            * conjdotxf;
				r1c.real += a1c.real * w0c.real - a1c.imag * w0c.imag
				            * conjdotxf;
				r1c.imag += a1c.imag * w0c.real + a1c.real * w0c.imag
				            * conjdotxf;
				r2c.real += a2c.real * w0c.real - a2c.imag * w0c.imag
				            * conjdotxf;
				r2c.imag += a2c.imag * w0c.real + a2c.real * w0c.imag
				            * conjdotxf;
				r3c.real += a3c.real * w0c.real - a3c.imag * w0c.imag
				            * conjdotxf;
				r3c.imag += a3c.imag * w0c.real + a3c.real * w0c.imag
				            * conjdotxf;

				rho[0] = r0c;
				rho[1] = r1c;
				rho[2] = r2c;
				rho[3] = r3c;

				// axpy
				scomplex z0c = z[p];

				z0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag
				            * conjaxpyf;
				z0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag
				            * conjaxpyf;
				z0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag
				            * conjaxpyf;
				z0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag
				            * conjaxpyf;
				z0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag
				            * conjaxpyf;
				z0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag
				            * conjaxpyf;
				z0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag
				            * conjaxpyf;
				z0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag
				            * conjaxpyf;

				z[p] = z0c;
			}
		}

		// Conjugating the final result if conjat
		if ( bli_is_conj( conjat ) )
		{
			for ( dim_t j = 0; j < 4; ++j )
			{
				PASTEMAC(c,conjs)( rho[j] );
			}
		}

		// Scaling the dot product result with alpha
		// and adding the result to vector y
		for ( dim_t j = 0; j < 4; ++j )
		{
			PASTEMAC(c,axpys)( *alpha, rho[j], y[j] );
		}
	}
	else
	{
		// For non-unit increments
		/* Query the context for the kernel function pointer. */
		const num_t dt     = PASTEMAC(c,type);
		PASTECH(c,dotxf_ker_ft) kfp_df 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx );
		PASTECH(c,axpyf_ker_ft) kfp_af 	=
			bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx );

		kfp_df
			(
			 conjat,
			 conjw,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 w, incw,
			 beta,
			 y, incy,
			 cntx
			);

		kfp_af
			(
			 conja,
			 conjx,
			 m,
			 b_n,
			 alpha,
			 a, inca, lda,
			 x, incx,
			 z, incz,
			 cntx
			);
	}
}
