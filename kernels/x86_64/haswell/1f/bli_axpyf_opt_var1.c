/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

typedef union
{
	__m256 v;
	float f[8];
}v8ff_t;

void bli_saxpyf_int_var1
     (
       conj_t  conja,
       conj_t  conjx,
       dim_t   m,
       dim_t   b_n,
       float* restrict alpha,
       float* restrict a, inc_t inca, inc_t lda,
       float* restrict x, inc_t incx,
       float* restrict y, inc_t incy,
       cntx_t* cntx
     )
{
	float* restrict a0;
	float* restrict a1;
	float* restrict a2;
	float* restrict a3;
	float* restrict a4;
	float* restrict a5;
	float* restrict a6;
	float* restrict a7;

	float  a0c, a1c, a2c, a3c;
	float  a4c, a5c, a6c, a7c;

	float* restrict y1;

	float  alpha_chi0;
	float  alpha_chi1;
	float  alpha_chi2;
	float  alpha_chi3;
	float  alpha_chi4;
	float  alpha_chi5;
	float  alpha_chi6;
	float  alpha_chi7;

	const dim_t fusefac = 8;
	const dim_t n_elem_per_reg = 8;


	dim_t m_run, m_left;
	int j;
	float  y_reg0;

	v8ff_t alpha0_vreg, alpha1_vreg, alpha2_vreg, alpha3_vreg;
	v8ff_t alpha4_vreg, alpha5_vreg, alpha6_vreg, alpha7_vreg;
	v8ff_t a0_vreg, a1_vreg, a2_vreg, a3_vreg;
	v8ff_t a4_vreg, a5_vreg, a6_vreg, a7_vreg;
	v8ff_t y_res_vreg;

    if ( bli_zero_dim2( m, b_n ) ) return;

	bool_t            use_ref = FALSE;

	if ( ( b_n != fusefac) || inca != 1 || incx != 1 || incy != 1 )
		use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		BLIS_SAXPYF_KERNEL_REF( conja, conjx, m, b_n, alpha, a, inca, lda, x, incx, y, incy, cntx );
		return;
	}

	m_run =  m / n_elem_per_reg;
	m_left = m % n_elem_per_reg;

	a0   = a + (0  )*lda;
	a1   = a + (1  )*lda;
	a2   = a + (2  )*lda;
	a3   = a + (3  )*lda;
	a4   = a + (4  )*lda;
	a5   = a + (5  )*lda;
	a6   = a + (6  )*lda;
	a7   = a + (7  )*lda;

	y1   = y;

	alpha_chi0        = *( x + 0 *incx );
	alpha_chi1        = *( x + 1 *incx );
	alpha_chi2        = *( x + 2 *incx );
	alpha_chi3        = *( x + 3 *incx );
	alpha_chi4        = *( x + 4 *incx );
	alpha_chi5        = *( x + 5 *incx );
	alpha_chi6        = *( x + 6 *incx );
	alpha_chi7        = *( x + 7 *incx );

	PASTEMAC2(s,s,scals)( *alpha, alpha_chi0 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi1 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi2 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi3 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi4 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi5 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi6 );
	PASTEMAC2(s,s,scals)( *alpha, alpha_chi7 );

	alpha0_vreg.v = _mm256_broadcast_ss(&(alpha_chi0));
	alpha1_vreg.v = _mm256_broadcast_ss(&(alpha_chi1));
	alpha2_vreg.v = _mm256_broadcast_ss(&(alpha_chi2));
	alpha3_vreg.v = _mm256_broadcast_ss(&(alpha_chi3));
	alpha4_vreg.v = _mm256_broadcast_ss(&(alpha_chi4));
	alpha5_vreg.v = _mm256_broadcast_ss(&(alpha_chi5));
	alpha6_vreg.v = _mm256_broadcast_ss(&(alpha_chi6));
	alpha7_vreg.v = _mm256_broadcast_ss(&(alpha_chi7));

	for(j = 0; j < m_run; j++)
	{

		y_res_vreg.v = _mm256_loadu_ps(y1);
		a0_vreg.v = _mm256_loadu_ps(a0);
		a1_vreg.v = _mm256_loadu_ps(a1);
		a2_vreg.v = _mm256_loadu_ps(a2);
		a3_vreg.v = _mm256_loadu_ps(a3);
		a4_vreg.v = _mm256_loadu_ps(a4);
		a5_vreg.v = _mm256_loadu_ps(a5);
		a6_vreg.v = _mm256_loadu_ps(a6);
		a7_vreg.v = _mm256_loadu_ps(a7);

		y_res_vreg.v += a0_vreg.v * alpha0_vreg.v;
		y_res_vreg.v += a1_vreg.v * alpha1_vreg.v;
		y_res_vreg.v += a2_vreg.v * alpha2_vreg.v;
		y_res_vreg.v += a3_vreg.v * alpha3_vreg.v;
		y_res_vreg.v += a4_vreg.v * alpha4_vreg.v;
		y_res_vreg.v += a5_vreg.v * alpha5_vreg.v;
		y_res_vreg.v += a6_vreg.v * alpha6_vreg.v;
		y_res_vreg.v += a7_vreg.v * alpha7_vreg.v;

		_mm256_storeu_ps(y1,y_res_vreg.v);

		y1 += n_elem_per_reg;
		a0 += n_elem_per_reg;
		a1 += n_elem_per_reg;
		a2 += n_elem_per_reg;
		a3 += n_elem_per_reg;
		a4 += n_elem_per_reg;
		a5 += n_elem_per_reg;
		a6 += n_elem_per_reg;
		a7 += n_elem_per_reg;

	}

	for(j = 0; j < m_left ; j++)
	{

		a0c = *a0;
		a1c = *a1;
		a2c = *a2;
		a3c = *a3;
		a4c = *a4;
		a5c = *a5;
		a6c = *a6;
		a7c = *a7;

		y_reg0 = (*y1);
		y_reg0 += alpha_chi0 * a0c;
		y_reg0 += alpha_chi1 * a1c;
		y_reg0 += alpha_chi2 * a2c;
		y_reg0 += alpha_chi3 * a3c;
		y_reg0 += alpha_chi4 * a4c;
		y_reg0 += alpha_chi5 * a5c;
		y_reg0 += alpha_chi6 * a6c;
		y_reg0 += alpha_chi7 * a7c;
		(*y1) = y_reg0;

		y1 += 1;
		a0 += 1;
		a1 += 1;
		a2 += 1;
		a3 += 1;
		a4 += 1;
		a5 += 1;
		a6 += 1;
		a7 += 1;
	}
}

typedef union
{
	__m256d v;
	double d[4];
}v4df_t;

void bli_daxpyf_int_var1
     (
       conj_t  conja,
       conj_t  conjx,
       dim_t   m,
       dim_t   b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* cntx
     )
{
	double* restrict a0;
	double* restrict a1;
	double* restrict a2;
	double* restrict a3;

	double* restrict y1;

	double  alpha_chi0;
	double  alpha_chi1;
	double  alpha_chi2;
	double  alpha_chi3;
	double  a0c, a1c, a2c, a3c;
	dim_t m_run, m_left;
	int  j, i;
	double  y_reg0;
	const dim_t fusefac = 4;
	const dim_t n_elem_per_reg = 4;

	v4df_t alpha0_vreg, alpha1_vreg, alpha2_vreg, alpha3_vreg;
	v4df_t a0_vreg, a1_vreg, a2_vreg, a3_vreg;
	v4df_t y_res_vreg;

    if ( bli_zero_dim2( m, b_n ) ) return;

	bool_t            use_ref = FALSE;

	if ( ( b_n < fusefac) || inca != 1 || incx != 1 || incy != 1 )
		use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{
		BLIS_DAXPYF_KERNEL_REF( conja, conjx, m, b_n, alpha, a, inca, lda, x, incx, y, incy, cntx );
		return;
	}

	m_run =  m / n_elem_per_reg;
	m_left = m % n_elem_per_reg;

	for(i = 0 ; i < b_n; i += 4)
	{
		a0   = a + (i + 0)*lda;
		a1   = a + (i + 1)*lda;
		a2   = a + (i + 2)*lda;
		a3   = a + (i + 3)*lda;

		y1   = y + (0  )*incy;

		alpha_chi0        = *( x + (i + 0) *incx );
		alpha_chi1        = *( x + (i + 1) *incx );
		alpha_chi2        = *( x + (i + 2) *incx );
		alpha_chi3        = *( x + (i + 3) *incx );

		PASTEMAC2(d,d,scals)( *alpha, alpha_chi0 );
		PASTEMAC2(d,d,scals)( *alpha, alpha_chi1 );
		PASTEMAC2(d,d,scals)( *alpha, alpha_chi2 );
		PASTEMAC2(d,d,scals)( *alpha, alpha_chi3 );

		alpha0_vreg.v = _mm256_broadcast_sd(&(alpha_chi0));
		alpha1_vreg.v = _mm256_broadcast_sd(&(alpha_chi1));
		alpha2_vreg.v = _mm256_broadcast_sd(&(alpha_chi2));
		alpha3_vreg.v = _mm256_broadcast_sd(&(alpha_chi3));

		for(j = 0; j < m_run; j++)
		{
			y_res_vreg.v = _mm256_loadu_pd(y1);
			a0_vreg.v = _mm256_loadu_pd(a0);
			a1_vreg.v = _mm256_loadu_pd(a1);
			a2_vreg.v = _mm256_loadu_pd(a2);
			a3_vreg.v = _mm256_loadu_pd(a3);

			y_res_vreg.v += a0_vreg.v * alpha0_vreg.v;
			y_res_vreg.v += a1_vreg.v * alpha1_vreg.v;
			y_res_vreg.v += a2_vreg.v * alpha2_vreg.v;
			y_res_vreg.v += a3_vreg.v * alpha3_vreg.v;

			_mm256_storeu_pd(y1,y_res_vreg.v);

			y1 += n_elem_per_reg;
			a0 += n_elem_per_reg;
			a1 += n_elem_per_reg;
			a2 += n_elem_per_reg;
			a3 += n_elem_per_reg;

		}

		for(j = 0; j < m_left ; j++)
		{
			a0c = *a0;
			a1c = *a1;
			a2c = *a2;
			a3c = *a3;

			y_reg0 = (*y1);
			y_reg0 += alpha_chi0 * a0c;
			y_reg0 += alpha_chi1 * a1c;
			y_reg0 += alpha_chi2 * a2c;
			y_reg0 += alpha_chi3 * a3c;

			(*y1) = y_reg0;

			y1 += 1;
			a0 += 1;
			a1 += 1;
			a2 += 1;
			a3 += 1;
		}
	}

}
