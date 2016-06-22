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
void bli_sdotxf_int_var1
     (
       conj_t          conjat,
       conj_t          conjx,
       dim_t           m,
       dim_t           b_n,
       float* restrict alpha,
       float* restrict a, inc_t inca, inc_t lda,
       float* restrict x, inc_t incx,
       float* restrict beta,
       float* restrict y, inc_t incy,
       cntx_t*         cntx
     )
{

	dim_t  i;
	bool_t use_ref = FALSE;
	const dim_t fusefac = 8;
	const dim_t n_elem_per_reg = 8;

	float*  restrict x0;
	float*  restrict x1;
	float*  restrict x2;
	float*  restrict x3;
	float*  restrict x4;
	float*  restrict x5;
	float*  restrict x6;
	float*  restrict x7;

	float*  restrict y0;
	float*  restrict y1;
	float*  restrict y2;
	float*  restrict y3;
	float*  restrict y4;
	float*  restrict y5;
	float*  restrict y6;
	float*  restrict y7;

	float   rho0, rho1, rho2, rho3;
	float   rho4, rho5, rho6, rho7;

	float   x0c, x1c, x2c, x3c, y0c;
	float   x4c, x5c, x6c, x7c;

	dim_t m_run, m_left;
	v8ff_t x0c_vec, x1c_vec, x2c_vec, x3c_vec;
	v8ff_t x4c_vec, x5c_vec, x6c_vec, x7c_vec;
	v8ff_t y0c_vec, rho0_vec, rho1_vec, rho2_vec, rho3_vec;
	v8ff_t rho4_vec, rho5_vec, rho6_vec, rho7_vec;

	if ( bli_zero_dim1( b_n ) ) return;

	// If the vector lengths are zero, scale r by beta and return.
	if ( bli_zero_dim1( m ) )
	{
		bli_sscalv( BLIS_NO_CONJUGATE,
		            b_n,
		            beta,
		            y, incy,
		            cntx );
		return;
	}

	if ( b_n != fusefac )
	{
		use_ref = TRUE;
	}
    else if ( inca != 1 || incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

	//use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{

		BLIS_SDOTXF_KERNEL_REF( conjat,
		                        conjx,
		                        m,
		                        b_n,
		                        alpha,
		                        a, inca, lda,
		                        x, incx,
		                        beta,
		                        y, incy,
		                        cntx );
		return;
	}

	x0 = a;
	x1 = a +   lda;
	x2 = a + 2*lda;
	x3 = a + 3*lda;
	x4 = a + 4*lda;
	x5 = a + 5*lda;
	x6 = a + 6*lda;
	x7 = a + 7*lda;

	y0 = x;

	PASTEMAC(d,set0s)( rho0 );
	PASTEMAC(d,set0s)( rho1 );
	PASTEMAC(d,set0s)( rho2 );
	PASTEMAC(d,set0s)( rho3 );

	m_run =  m / n_elem_per_reg;
	m_left = m % n_elem_per_reg;

	rho0_vec.v = _mm256_setzero_ps();
	rho1_vec.v = _mm256_setzero_ps();
	rho2_vec.v = _mm256_setzero_ps();
	rho3_vec.v = _mm256_setzero_ps();
	rho4_vec.v = _mm256_setzero_ps();
	rho5_vec.v = _mm256_setzero_ps();
	rho6_vec.v = _mm256_setzero_ps();
	rho7_vec.v = _mm256_setzero_ps();


	for(i = 0; i < m_run; i++)
	{

		x0c_vec.v = _mm256_loadu_ps(x0);
		x1c_vec.v = _mm256_loadu_ps(x1);
		x2c_vec.v = _mm256_loadu_ps(x2);
		x3c_vec.v = _mm256_loadu_ps(x3);
		x4c_vec.v = _mm256_loadu_ps(x4);
		x5c_vec.v = _mm256_loadu_ps(x5);
		x6c_vec.v = _mm256_loadu_ps(x6);
		x7c_vec.v = _mm256_loadu_ps(x7);

		y0c_vec.v = _mm256_loadu_ps(y0);

		rho0_vec.v += x0c_vec.v * y0c_vec.v;
		rho1_vec.v += x1c_vec.v * y0c_vec.v;
		rho2_vec.v += x2c_vec.v * y0c_vec.v;
		rho3_vec.v += x3c_vec.v * y0c_vec.v;
		rho4_vec.v += x4c_vec.v * y0c_vec.v;
		rho5_vec.v += x5c_vec.v * y0c_vec.v;
		rho6_vec.v += x6c_vec.v * y0c_vec.v;
		rho7_vec.v += x7c_vec.v * y0c_vec.v;

		x0 += n_elem_per_reg;
		x1 += n_elem_per_reg;
		x2 += n_elem_per_reg;
		x3 += n_elem_per_reg;
		x4 += n_elem_per_reg;
		x5 += n_elem_per_reg;
		x6 += n_elem_per_reg;
		x7 += n_elem_per_reg;
		y0 += n_elem_per_reg;

	}

	rho0 = rho0_vec.f[0] + rho0_vec.f[1] + rho0_vec.f[2] + rho0_vec.f[3] +
		   rho0_vec.f[4] + rho0_vec.f[5] + rho0_vec.f[6] + rho0_vec.f[7];
	rho1 = rho1_vec.f[0] + rho1_vec.f[1] + rho1_vec.f[2] + rho1_vec.f[3] +
		   rho1_vec.f[4] + rho1_vec.f[5] + rho1_vec.f[6] + rho1_vec.f[7];
	rho2 = rho2_vec.f[0] + rho2_vec.f[1] + rho2_vec.f[2] + rho2_vec.f[3] +
		   rho2_vec.f[4] + rho2_vec.f[5] + rho2_vec.f[6] + rho2_vec.f[7];
	rho3 = rho3_vec.f[0] + rho3_vec.f[1] + rho3_vec.f[2] + rho3_vec.f[3] +
		   rho3_vec.f[4] + rho3_vec.f[5] + rho3_vec.f[6] + rho3_vec.f[7];


	rho4 = rho4_vec.f[0] + rho4_vec.f[1] + rho4_vec.f[2] + rho4_vec.f[3] +
		   rho4_vec.f[4] + rho4_vec.f[5] + rho4_vec.f[6] + rho4_vec.f[7];
	rho5 = rho5_vec.f[0] + rho5_vec.f[1] + rho5_vec.f[2] + rho5_vec.f[3] +
		   rho5_vec.f[4] + rho5_vec.f[5] + rho5_vec.f[6] + rho5_vec.f[7];
	rho6 = rho6_vec.f[0] + rho6_vec.f[1] + rho6_vec.f[2] + rho6_vec.f[3] +
		   rho6_vec.f[4] + rho6_vec.f[5] + rho6_vec.f[6] + rho6_vec.f[7];
	rho7 = rho7_vec.f[0] + rho7_vec.f[1] + rho7_vec.f[2] + rho7_vec.f[3] +
		   rho7_vec.f[4] + rho7_vec.f[5] + rho7_vec.f[6] + rho7_vec.f[7];


	for(i = 0; i < m_left; i++)
	{
		x0c = *x0;
		x1c = *x1;
		x2c = *x2;
		x3c = *x3;
		x4c = *x4;
		x5c = *x5;
		x6c = *x6;
		x7c = *x7;

		y0c = *y0;

		rho0 += x0c * y0c;
		rho1 += x1c * y0c;
		rho2 += x2c * y0c;
		rho3 += x3c * y0c;
		rho4 += x4c * y0c;
		rho5 += x5c * y0c;
		rho6 += x6c * y0c;
		rho7 += x7c * y0c;

		x0 += 1;
		x1 += 1;
		x2 += 1;
		x3 += 1;
		x4 += 1;
		x5 += 1;
		x6 += 1;
		x7 += 1;
		y0 += 1;
	}

	y0 = y;
	y1 = y0 + 1;
	y2 = y1 + 1;
	y3 = y2 + 1;
	y4 = y3 + 1;
	y5 = y4 + 1;
	y6 = y5 + 1;
	y7 = y6 + 1;

	(*y0) = (*y0) * (*beta) + rho0 * (*alpha);
	(*y1) = (*y1) * (*beta) + rho1 * (*alpha);
	(*y2) = (*y2) * (*beta) + rho2 * (*alpha);
	(*y3) = (*y3) * (*beta) + rho3 * (*alpha);
	(*y4) = (*y4) * (*beta) + rho4 * (*alpha);
	(*y5) = (*y5) * (*beta) + rho5 * (*alpha);
	(*y6) = (*y6) * (*beta) + rho6 * (*alpha);
	(*y7) = (*y7) * (*beta) + rho7 * (*alpha);

}



typedef union
{
	__m256d v;
	double d[4];
}v4df_t;

void bli_ddotxf_int_var1
     (
       conj_t          conjat,
       conj_t          conjx,
       dim_t           m,
       dim_t           b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict beta,
       double* restrict y, inc_t incy,
       cntx_t*         cntx
     )
{

	double*  restrict alpha_cast = alpha;
	double*  restrict beta_cast = beta;
	double*  restrict a_cast = a;
	double*  restrict x_cast = x;
	double*  restrict y_cast = y;

	double*  restrict x0;
	double*  restrict x1;
	double*  restrict x2;
	double*  restrict x3;

	double*  restrict y0;
	double*  restrict y1;
	double*  restrict y2;
	double*  restrict y3;

	double            rho0, rho1, rho2, rho3;
	double            x0c, x1c, x2c, x3c, y0c;
	dim_t  i;
	bool_t            use_ref = FALSE;
	const dim_t fusefac = 4;
	const dim_t n_elem_per_reg = 4;

	dim_t m_run, m_left;
	v4df_t x0c_vec, x1c_vec, x2c_vec, x3c_vec;
	v4df_t y0c_vec, rho0_vec, rho1_vec, rho2_vec, rho3_vec;

	if ( bli_zero_dim1( b_n ) ) return;

	// If the vector lengths are zero, scale r by beta and return.
	if ( bli_zero_dim1( m ) )
	{
		bli_dscalv( BLIS_NO_CONJUGATE,
		            b_n,
		            beta_cast,
		            y_cast, incy,
		            cntx );
		return;
	}

    // If there is anything that would interfere with our use of aligned
    // vector loads/stores, call the reference implementation.
	if ( b_n != fusefac )
	{
		use_ref = TRUE;
	}
    else if ( inca != 1 || incx != 1 || incy != 1 )
    {
        use_ref = TRUE;
    }

	//use_ref = TRUE;
	// Call the reference implementation if needed.
	if ( use_ref == TRUE )
	{

		BLIS_DDOTXF_KERNEL_REF( conjat,
		                        conjx,
		                        m,
		                        b_n,
		                        alpha_cast,
		                        a_cast, inca, lda,
		                        x_cast, incx,
		                        beta_cast,
		                        y_cast, incy,
		                        cntx );
		return;
	}

	x0 = a_cast;
	x1 = a_cast +   lda;
	x2 = a_cast + 2*lda;
	x3 = a_cast + 3*lda;
	y0 = x_cast;

	PASTEMAC(d,set0s)( rho0 );
	PASTEMAC(d,set0s)( rho1 );
	PASTEMAC(d,set0s)( rho2 );
	PASTEMAC(d,set0s)( rho3 );

	m_run =  m / n_elem_per_reg;
	m_left = m % n_elem_per_reg;

	rho0_vec.v = _mm256_setzero_pd();
	rho1_vec.v = _mm256_setzero_pd();
	rho2_vec.v = _mm256_setzero_pd();
	rho3_vec.v = _mm256_setzero_pd();

	for(i = 0; i < m_run; i++)
	{

		x0c_vec.v = _mm256_loadu_pd(x0);
		x1c_vec.v = _mm256_loadu_pd(x1);
		x2c_vec.v = _mm256_loadu_pd(x2);
		x3c_vec.v = _mm256_loadu_pd(x3);

		y0c_vec.v = _mm256_loadu_pd(y0);

		rho0_vec.v += x0c_vec.v * y0c_vec.v;
		rho1_vec.v += x1c_vec.v * y0c_vec.v;
		rho2_vec.v += x2c_vec.v * y0c_vec.v;
		rho3_vec.v += x3c_vec.v * y0c_vec.v;

		x0 += n_elem_per_reg;
		x1 += n_elem_per_reg;
		x2 += n_elem_per_reg;
		x3 += n_elem_per_reg;
		y0 += n_elem_per_reg;

	}

	rho0 += rho0_vec.d[0] + rho0_vec.d[1] + rho0_vec.d[2] + rho0_vec.d[3];
	rho1 += rho1_vec.d[0] + rho1_vec.d[1] + rho1_vec.d[2] + rho1_vec.d[3];
	rho2 += rho2_vec.d[0] + rho2_vec.d[1] + rho2_vec.d[2] + rho2_vec.d[3];
	rho3 += rho3_vec.d[0] + rho3_vec.d[1] + rho3_vec.d[2] + rho3_vec.d[3];

	for(i = 0; i < m_left; i++)
	{

		x0c = *x0;
		x1c = *x1;
		x2c = *x2;
		x3c = *x3;
		y0c = *y0;

		rho0 += x0c * y0c;
		rho1 += x1c * y0c;
		rho2 += x2c * y0c;
		rho3 += x3c * y0c;

		x0 += 1;
		x1 += 1;
		x2 += 1;
		x3 += 1;
		y0 += 1;
	}

	y0 = y_cast;
	y1 = y0 + 1;
	y2 = y1 + 1;
	y3 = y2 + 1;

	(*y0) = (*y0) * (*beta_cast) + rho0 * (*alpha_cast);
	(*y1) = (*y1) * (*beta_cast) + rho1 * (*alpha_cast);
	(*y2) = (*y2) * (*beta_cast) + rho2 * (*alpha_cast);
	(*y3) = (*y3) * (*beta_cast) + rho3 * (*alpha_cast);
}
