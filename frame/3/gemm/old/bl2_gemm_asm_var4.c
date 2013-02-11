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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                           void*   b, inc_t rs_b, inc_t cs_b,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T ftypes[BLIS_NUM_FP_TYPES] =
{
	bl2_sgemm_asm_var4,
	bl2_cgemm_asm_var4,
	bl2_dgemm_asm_var4,
	bl2_zgemm_asm_var4
};

void bl2_gemm_asm_var4( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	num_t     dt_exec   = bl2_obj_execution_datatype( *c );
	//num_t     dt_a      = bl2_obj_datatype( *a );
	//num_t     dt_b      = bl2_obj_datatype( *b );

	dim_t     m         = bl2_obj_length( *c );
	dim_t     n         = bl2_obj_width( *c );
	dim_t     k         = bl2_obj_width( *a );

	void*     buf_a     = bl2_obj_buffer_at_off( *a );
	inc_t     rs_a      = bl2_obj_row_stride( *a );
	inc_t     cs_a      = bl2_obj_col_stride( *a );
	inc_t     ps_a      = bl2_obj_panel_stride( *a );

	void*     buf_b     = bl2_obj_buffer_at_off( *b );
	inc_t     rs_b      = bl2_obj_row_stride( *b );
	inc_t     cs_b      = bl2_obj_col_stride( *b );

	void*     buf_c     = bl2_obj_buffer_at_off( *c );
	inc_t     rs_c      = bl2_obj_row_stride( *c );
	inc_t     cs_c      = bl2_obj_col_stride( *c );

	FUNCPTR_T f;

	// Handle the special case where c and a are complex and b is real.
	// Note that this is the ONLY case allowed by the inner kernel whereby
	// the datatypes of a and b differ. In this situation, the execution
	// datatype is real, so we need to inflate the m and leading dimensions
	// by a factor of two.
/*
	if ( bl2_is_complex( dt_a ) && bl2_is_real( dt_b ) )
	{
		m    *= 2;
		cs_c *= 2;
		cs_a *= 2;
	}
*/

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( m,
	   n,
	   k,
	   buf_a, rs_a, cs_a, ps_a,
	   buf_b, rs_b, cs_b,
	   buf_c, rs_c, cs_c );
}

void PASTEMAC(s,gemm_asm_var4)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

void PASTEMAC(c,gemm_asm_var4)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

void PASTEMAC(z,gemm_asm_var4)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

#include "pmmintrin.h"

typedef union
{
    __m128d v;
    double  d[2];
} v2df_t;

#define NOSSE 0

void PASTEMAC(d,gemm_asm_var4)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{
	double* restrict a_cast = a;
	double* restrict b_cast = b;
	double* restrict c_cast = c;
	double* restrict a1;
	double* restrict b1;
	double* restrict c1;
	double* restrict a11;
	double* restrict b11;
	double* restrict c11;

	double* restrict alpha00;
	double* restrict alpha20;
	double* restrict beta00;
	double* restrict beta01;

	double* restrict gamma00;
	double* restrict gamma20;
	double* restrict gamma01;
	double* restrict gamma21;

	v2df_t  c00v, c01v;
	v2df_t  c10v, c11v;
	v2df_t  a0v, a1v;
	v2df_t  b0v, b1v;

	dim_t   i, j, h;

	dim_t   n_iter = n / 2;
	dim_t   n_left = n % 2;

	dim_t   m_iter = m / 4;
	dim_t   m_left = m % 4;

	//dim_t   k_iter = k / 2;
	//dim_t   k_left = k % 2;
	dim_t   k_iter = k / 2;
	dim_t   k_left = k % 2;


	b1 = b_cast;
	c1 = c_cast;

	for ( j = 0; j < n_iter; ++j )
	{
		a1  = a_cast;
		c11 = c1;

		gamma00 = c11;
		gamma20 = c11 + 2;

		gamma01 = c11     + cs_c;
		gamma21 = c11 + 2 + cs_c;

		for ( i = 0; i < m_iter; ++i )
		{
/*
			gamma00 = c11 + 0*rs_c + 0*cs_c;
			gamma20 = c11 + 2*rs_c + 0*cs_c;

			gamma01 = c11 + 0*rs_c + 1*cs_c;
			gamma21 = c11 + 2*rs_c + 1*cs_c;
*/

			a11 = a1;
			b11 = b1;

			c00v.v = _mm_load_pd( gamma00 );
			c10v.v = _mm_load_pd( gamma20 );
			c01v.v = _mm_load_pd( gamma01 );
			c11v.v = _mm_load_pd( gamma21 );

			alpha00 = a11;
			alpha20 = a11 + 2;

			beta00  = b11;
			beta01  = b11 + cs_b;

			for ( h = 0; h < k_iter; ++h )
			{

				a0v.v = _mm_load_pd( alpha00 );
				a1v.v = _mm_load_pd( alpha20 );
				alpha00 += 4;
				alpha20 += 4;

				b0v.v = _mm_loaddup_pd( beta00 );
				beta00 += 1;
				c00v.v += a0v.v * b0v.v;
				c10v.v += a1v.v * b0v.v;

				b1v.v = _mm_loaddup_pd( beta01 );
				beta01 += 1;
				c01v.v += a0v.v * b1v.v;
				c11v.v += a1v.v * b1v.v;


				a0v.v = _mm_load_pd( alpha00 );
				a1v.v = _mm_load_pd( alpha20 );
				alpha00 += 4;
				alpha20 += 4;

				b0v.v = _mm_loaddup_pd( beta00 );
				beta00 += 1;
				c00v.v += a0v.v * b0v.v;
				c10v.v += a1v.v * b0v.v;

				b1v.v = _mm_loaddup_pd( beta01 );
				beta01 += 1;
				c01v.v += a0v.v * b1v.v;
				c11v.v += a1v.v * b1v.v;



				//alpha00 += 8;
				//alpha20 += 8;


			}

			for ( h = 0; h < k_left; ++h )
			{
				a0v.v = _mm_load_pd( alpha00 );
				a1v.v = _mm_load_pd( alpha20 );

				b0v.v = _mm_loaddup_pd( beta00++ );
				c00v.v += a0v.v * b0v.v;
				c10v.v += a1v.v * b0v.v;

				b1v.v = _mm_loaddup_pd( beta01++ );
				c01v.v += a0v.v * b1v.v;
				c11v.v += a1v.v * b1v.v;

				alpha00 += 4;
				alpha20 += 4;
			}

			_mm_store_pd( gamma00, c00v.v );
			_mm_store_pd( gamma20, c10v.v );
			_mm_store_pd( gamma01, c01v.v );
			_mm_store_pd( gamma21, c11v.v );

			//a1  += 4*rs_a;
			//c11 += 4*rs_c;
			//a1  += 4;
/*
			a1  += ps_a;
			c11 += 4;
*/
			gamma00 += 4;
			gamma20 += 4;
			gamma01 += 4;
			gamma21 += 4;
		}
/*
		for ( i = 0; i < m_left; ++i )
		{
		}
*/
		b1 += 2*cs_b;
		c1 += 2*cs_c;
	}
/*
	for ( j = 0; j < n_left ++j )
	{
	}
*/


}
