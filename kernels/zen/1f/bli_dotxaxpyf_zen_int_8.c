/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Advanced Micro Devices, Inc. All rights reserved.

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
	if ((inca == 1) && (incw == 1) && (incx == 1)
			&& (incy == 1) && (incz == 1) && (b_n == 8))
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
